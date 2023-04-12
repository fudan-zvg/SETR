import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint

import torch.utils.checkpoint as checkpoint
from ..backbones.region_attention import LayerNorm2d
import math

from mmseg.models.backbones.hlg_share import AttentionShareGlobal, AttentionShareLocal, DWMlp, InnerPatchEmbed

from ..builder import BACKBONES


NEG_INF = -1000000


def conv_3x3_bn(inp, oup, down_sample=False):
    stride = 1 if down_sample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class InnerPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, input_dim, embed_dim, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        if down_sample:
            self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, kernel_size=3, stride=2, padding=1, groups=input_dim),
                                        nn.BatchNorm2d(embed_dim),
                                        nn.GELU())

    def forward(self, x, H, W):
        if self.down_sample:
            B, N, C = x.shape
            assert N == H * W
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.proj(x)
            # print(x.size(), self.H, self.W)
            x = x.flatten(2).transpose(1, 2)
        return x

class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., down_sample=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.down_sample = down_sample
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        stride = 2 if down_sample else 1
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, groups=hidden_features, stride=stride, padding=1)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.fc3 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class HLGAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, dynamic_position_bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.dynamic_position_bias = dynamic_position_bias

        if self.dynamic_position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.norm = nn.LayerNorm(dim)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x, H=None, W=None, h0_token=None, group_size=(7,7)):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if h0_token is not None:
            x_ = h0_token
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print("attn.shape={}".format(attn.shape))

        # attn + B
        if self.dynamic_position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
                
            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            # print("coords_flatten.shape={}".format(coords_flatten.shape))
            # temp = input("enter")

            if len(group_size) > 2:
                coords_h_0 = torch.arange(group_size[2], device=attn.device)
                coords_w_0 = torch.arange(group_size[3], device=attn.device)
                coords_0 = torch.stack(torch.meshgrid([coords_h_0, coords_w_0]))  # 2, Gh, Gw
                coords_flatten_0 = torch.flatten(coords_0, 1)  # 2, Gh*Gw
                relative_coords = coords_flatten[:, :, None] - coords_flatten_0[:, None, :]
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
                relative_coords[:, :, 0] += group_size[2] - 1  # shift to start from 0
                relative_coords[:, :, 1] += group_size[3] - 1
                relative_coords[:, :, 0] *= 2 * group_size[3] - 1 # TODO
                relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

                pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads

                # select position bias
                relative_position_bias = pos[relative_position_index.view(-1)].view( 
                    group_size[0] * group_size[1], group_size[2] * group_size[3], -1)  # Gh*Gw,Gh*Gw,nH
                
            else:
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
                relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += group_size[1] - 1
                relative_coords[:, :, 0] *= 2 * group_size[1] - 1
                relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

                pos = self.pos(biases) # 2Gh-1 * 2Gw-1, heads

                # select position bias
                relative_position_bias = pos[relative_position_index.view(-1)].view( 
                    group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH

            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        """
        [N, C] * [C, N] = [N, N]
        Softmax([N, _N_])
        Softmax([N
        """
        return x


class HLGHeadLayers(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, h0_att=True,
                 transform_method='mean', reuse=True, h0_h1_fusion_type='attn', window_size=7, lsda_flag=0, interval=8,
                 dynamic_position_bias=False):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.h0_h1_fusion_type = h0_h1_fusion_type
        # lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        self.lsda_flag = lsda_flag
        self.window_size = window_size
        self.interval = interval # 8 4 2 1 按照分类的分辨率得到的interval
        self.dynamic_position_bias = dynamic_position_bias
        
        if self.interval != 1:
            self.norm0 = norm_layer(dim)
        
        self.norm1 = norm_layer(dim)
        self.attn = HLGAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, dynamic_position_bias=dynamic_position_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.transform_method = transform_method
        self.h0_att = h0_att
        if self.h0_att:
            self.normh0 = norm_layer(dim)
        self.repeat = 1
        
        if transform_method == 'hada':
            self.q = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
        elif transform_method == 'conv':
            self.h0_conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size)
            # self.h0_norm = norm_layer(dim)
        elif transform_method == 'dwconv':
            self.h0_conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size, groups=dim) # window_size
            # self.h0_norm = norm_layer(dim)
        
        self.spatial_smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, dim//4, kernel_size=1),
                                                           nn.BatchNorm2d(dim//4), nn.GELU(),
                                                           nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, groups=1),
                                                           nn.BatchNorm2d(dim//4), nn.GELU(),
                                                           nn.Conv2d(dim//4, dim, kernel_size=1),
                                                           nn.BatchNorm2d(dim)) for i in range(self.repeat)])

    def forward(self, inputs):
        """
        h0_pos: [1, c, h, w]
        """
        x, H, W, token, h_conf, w_conf, h0_pos, stage_index = inputs
        
        b, N, c = x.shape
        h0_token = None

        if stage_index == 4:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W, h0_token, group_size=(H, W)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        
        # padding
        x = x.contiguous().view(b, H, W, c).contiguous()
        
        size_div = self.interval if self.lsda_flag == 1 else self.window_size
        pad_l = pad_t = 0
        pad_r = (size_div - W % size_div) % size_div
        pad_b = (size_div - H % size_div) % size_div
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1
            
        if self.dilated_flag == 0:
            # plain local attnetion
            h1_x = x
            h1_x = h1_x.permute(0, 3, 1, 2).contiguous() # (b, c, H, W)
            for i in range(self.repeat):
                h1_x = nn.GELU()(h1_x + self.spatial_smooth[i](h1_x))

            # h1_x = unfold_layer(h1_x).transpose(1, 2)  # [b, hw, c*scale*scale]
            h1_x = h1_x.permute(0, 2, 3, 1).contiguous() # b H W c
            G = Gh = Gw = self.window_size
            h1_x = h1_x.reshape(b, Hp // G, G, Wp // G, G, c).permute(0, 1, 3, 2, 4, 5).contiguous()
            h1_x = h1_x.reshape(b * Hp * Wp // G**2, G**2, c)
            nG = Hp * Wp // G**2

            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hp // G, G, Wp // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nG, 1, G * G)
                attn_mask = torch.zeros((nG, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

            h1_x = h1_x + self.drop_path(self.attn(self.norm0(h1_x), group_size=(Gh, Gw)))

            if self.transform_method == 'mean':
                h0_token = h1_x.mean(dim=1)
            elif self.transform_method == 'max':
                h0_token = h1_x.max(dim=1)[0]
            elif self.transform_method == 'conv' or self.transform_method == 'dwconv':
                h1_x = h1_x.transpose(1, 2).contiguous().reshape(b * Hp * Wp // G**2, c, G, G)
                h0_token = self.h0_conv(h1_x)
                h1_x = h1_x.reshape(b * Hp * Wp // G**2, c, G * G).transpose(1, 2).contiguous()
            elif self.transform_method == 'hada':
                q = self.q(h1_x)
                v = self.v(h1_x)
                h0_token = q * v
                h0_token = h0_token.sum(dim=1)
            
            h0_token = h0_token.reshape(b, Hp * Wp // G**2, c)
            if self.h0_att:
                # h0_token = self.h0_block(h0_token)
                h0_token = h0_token + self.drop_path(self.attn(self.normh0(h0_token)))
            h1_feature = h1_x # b * newH * newW, kH * kW, c

            x = h1_feature.reshape(b, Hp // G, Wp // G, G, G, c).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C

        elif self.dilated_flag == 1:
            # dilated local attention
            h1_x = x
            h1_x = h1_x.permute(0, 3, 1, 2).contiguous() # (b, c, H, W)
            for i in range(self.repeat):
                h1_x = nn.GELU()(h1_x + self.spatial_smooth[i](h1_x))

            h1_x = h1_x.permute(0, 2, 3, 1).contiguous() # b H W c

            I, Gh, Gw = self.interval, Hp // self.interval, Wp // self.interval
            h1_x = h1_x.reshape(b, Gh, I, Gw, I, c).permute(0, 2, 4, 1, 3, 5).contiguous()
            h1_x = h1_x.reshape(b * I * I, Gh * Gw, c)
            nG = I ** 2
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nG, 1, Gh * Gw)
                attn_mask = torch.zeros((nG, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        
            h1_x = h1_x + self.drop_path(self.attn(self.norm0(h1_x), group_size=(Gh, Gw))) # (b * I * I, Gh * Gw, c)

            if self.transform_method == 'mean':
                h0_token = h1_x.mean(dim=1)
            elif self.transform_method == 'max':
                h0_token = h1_x.max(dim=1)[0]
            elif self.transform_method == 'conv' or self.transform_method == 'dwconv':
                h1_x = h1_x.transpose(1, 2).contiguous().reshape(b * I * I, c, Gh, Gw)
                h0_token = self.h0_conv(h1_x)
                h1_x = h1_x.reshape(b * I * I, c, Gh * Gw).transpose(1, 2).contiguous()
            elif self.transform_method == 'hada':
                q = self.q(h1_x)
                v = self.v(h1_x)
                h0_token = q * v
                h0_token = h0_token.sum(dim=1)

            h0_token = h0_token.reshape(b, I*I, c)
            if self.h0_att:
                h0_token = h0_token + self.drop_path(self.attn(self.normh0(h0_token)))
        
            h1_feature = h1_x  # b * newH * newW, kH * kW, c
            x = h1_feature.reshape(b, I, I, Gh, Gw, c).permute(0, 3, 1, 4, 2, 5).contiguous()

        x = x.reshape(b, Hp, Wp, c)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(b, H * W, c).contiguous()
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, h0_token, group_size=(H, W, Hp // Gh, Wp // Gw)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, kernel_size=7, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if kernel_size == 7:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(32), nn.GELU(),
                                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32), nn.GELU(),
                                      nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(embed_dim), nn.GELU())
        else:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=int(kernel_size//2), groups=in_chans),
                                      nn.BatchNorm2d(embed_dim),
                                      nn.GELU())

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
