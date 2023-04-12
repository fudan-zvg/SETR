import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint

import torch.utils.checkpoint as checkpoint
import math

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


class AttentionShareGlobal(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, dynamic_position_bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.dynamic_position_bias = True
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, q, k, v, dpb_pos, H=None, W=None, group_size=(7,7)):
        B, N, C = q.shape
        _, M, _ = k.shape

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attn + B
        if self.dynamic_position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0]).cuda()
            position_bias_w = torch.arange(1 - group_size[0], group_size[0]).cuda()
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            # self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(group_size[0]).cuda()
            coords_w = torch.arange(group_size[0]).cuda() # 1
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[0] - 1 # 1
            relative_coords[:, :, 0] *= 2 * group_size[0] - 1 # 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # relative_position_index = relative_position_index[:, 0:group_size[1]*group_size[1]]

            pos = dpb_pos(biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[0], group_size[0] * group_size[0], -1)  # Wh*Ww,Wh*Ww,nH
            # print("line 339: relative_position_bias.shape={}".format(relative_position_bias.shape)) # 3136 3136 2
            # print("group_size=", group_size)
            if group_size[0] != group_size[1]:
                # temp = relative_position_index
                relative_position_bias = relative_position_bias[:, 0:(group_size[1] * group_size[1]), :] # 3136 64 2
            # print("line 342: relative_position_bias.shape={}".format(relative_position_bias.shape))
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # print("line 344: attn.shape={} position_bias.shape={}".format(attn.shape, relative_position_bias.shape))
            if H == 7 and W == 7:
                attn[:, :, 0:49, 0:49] = attn[:, :, 0:49, 0:49] + relative_position_bias.unsqueeze(0)
            else:
                attn = attn + relative_position_bias.unsqueeze(0)
            # attn = attn + relative_position_bias.unsqueeze(0)
            # print("sum over")
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # print("attn.shape={} v.shape={} B={} N={} C={}".format(attn.shape, v.shape, B, N, C))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # print("x.shape={}".format(x.shape))
        
        x = self.proj(x)
        x = self.proj_drop(x)
        """
        [N, C] * [C, N] = [N, N]
        Softmax([N, _N_])
        Softmax([N

        """
        # print("x.shape2={}".format(x.shape))
        return x


class AttentionShareLocal(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, dynamic_position_bias=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.dynamic_position_bias = True

        self.attn_drop = nn.Dropout(attn_drop)
        

    def forward(self, q, k, v, dpb_pos, H=None, W=None, group_size=(7,7)):
        B, N, C = q.shape
        # print("forward start: H={} W={}".format(H, W))

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print("k v shape:", k.shape, v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # attn + B
        if self.dynamic_position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0]).cuda()
            position_bias_w = torch.arange(1 - group_size[0], group_size[0]).cuda()
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            # self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(group_size[0]).cuda()
            coords_w = torch.arange(group_size[0]).cuda() # 1
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[0] - 1 # 1
            relative_coords[:, :, 0] *= 2 * group_size[0] - 1 # 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            # relative_position_index = relative_position_index[:, 0:group_size[1]*group_size[1]]
            pos = dpb_pos(biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[0], group_size[0] * group_size[0], -1)  # Wh*Ww,Wh*Ww,nH
            # print("line 339: relative_position_bias.shape={}".format(relative_position_bias.shape)) # 3136 3136 2
            # print("group_size=", group_size)
            if group_size[0] != group_size[1]:
                # temp = relative_position_index
                relative_position_bias = relative_position_bias[:, 0:(group_size[1] * group_size[1]), :] # 3136 64 2
            # print("line 342: relative_position_bias.shape={}".format(relative_position_bias.shape))
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # print("line 344: attn.shape={} position_bias.shape={}".format(attn.shape, relative_position_bias.shape))
            if H == 7 and W == 7:
                attn[:, :, 0:49, 0:49] = attn[:, :, 0:49, 0:49] + relative_position_bias.unsqueeze(0)
            else:
                attn = attn + relative_position_bias.unsqueeze(0)
            # attn = attn + relative_position_bias.unsqueeze(0)
            # print("sum over")
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class HLGLayers(nn.Module):
    def __init__(self, input_dim, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, h0_att=True,
                 transform_method='mean', reuse=True, h0_h1_fusion_type='attn', window_size=7, lsda_flag=0, interval=8,
                 dynamic_position_bias=False, down_sample=False,):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.h0_h1_fusion_type = h0_h1_fusion_type
        # lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        self.lsda_flag = lsda_flag
        self.window_size = window_size
        self.interval = interval # 8 4 2 1 按照分类的分辨率得到的interval
        self.dynamic_position_bias = dynamic_position_bias
        
        if self.interval != 1:
            self.aux_q = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
            self.norm0 = norm_layer(dim)
            self.h0_norm = norm_layer(dim)
            self.pos_local = DynamicPosBias(dim // 4, num_heads, residual=False)
            self.local_attn = AttentionShareLocal(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=1, dynamic_position_bias=dynamic_position_bias)
            self.repeat = 1
            self.spatial_smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(dim, dim//4, kernel_size=1),
                                                            nn.BatchNorm2d(dim//4), nn.GELU(),
                                                            nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, groups=1),
                                                            nn.BatchNorm2d(dim//4), nn.GELU(),
                                                            nn.Conv2d(dim//4, dim, kernel_size=1),
                                                            nn.BatchNorm2d(dim)) for i in range(self.repeat)])
        self.down_sample = down_sample
        self.down_proj = InnerPatchEmbed(input_dim=input_dim, embed_dim=dim, down_sample=down_sample)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.pos_global = DynamicPosBias(dim // 4, num_heads, residual=False)
        self.global_attn = AttentionShareGlobal(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, dynamic_position_bias=dynamic_position_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = DWMlp(in_features=input_dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop, down_sample=down_sample)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.transform_method = transform_method

        if transform_method == 'hada':
            self.q = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
        elif transform_method == 'conv':
            self.h0_conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size)
            # self.h0_norm = norm_layer(dim)
        elif transform_method == 'dwconv':
            self.h0_conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=window_size, groups=dim) # window_size
            # self.h0_norm = norm_layer(dim)

    def forward(self, inputs):
        """
        h0_pos: [1, c, h, w]
        """
        x, H, W, h_conf, w_conf = inputs
        h0_token = None
        x = self.down_proj(x, H, W) + self.drop_path(self.norm2(self.mlp(x, H, W)))
        if self.down_sample:
            H = H // 2
            W = W // 2
        b, N, c = x.shape
        if self.interval == 1:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            x = x + self.drop_path(self.norm1(self.global_attn(q, k, v, self.pos_global, H, W, pdb_check=pdb_check)))
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

            share_q = self.q(self.norm0(h1_x))
            local_v = self.v(self.norm0(h1_x))

            h1_x_p = self.local_attn(share_q, share_q, local_v, self.pos_local, group_size=(Gh, Gw))
            h1_x = h1_x + h1_x_p    # b * newH * newW, kH * kW, c
            
            h0_token = h1_x.mean(dim=1)
            h0_token = h0_token.reshape(b, Hp * Wp // G**2, c)
            h0_token = self.h0_norm(h0_token)

            tmp_x = torch.cat([h1_x, share_q, h1_x_p], dim=0)
            tmp_x = tmp_x.reshape(3*b, Hp // G, Wp // G, G, G, c).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
        elif self.dilated_flag == 1:
            # dilated local attention
            h1_x = x
            h1_x = h1_x.permute(0, 3, 1, 2).contiguous() # (b, c, H, W)
            for i in range(self.repeat):
                h1_x = nn.GELU()(h1_x + self.spatial_smooth[i](h1_x))

            h1_x = h1_x.permute(0, 2, 3, 1).contiguous() # b H W c
            G = Gh = Gw = self.window_size
            h1_x = h1_x.reshape(b, G, Hp // G, G, Wp // G, c).permute(0, 2, 4, 1, 3, 5).contiguous()
            h1_x = h1_x.reshape(b * Hp * Wp // G**2, G**2, c)
            nG = Hp * Wp // G**2

            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, G, Hp // G, G, Wp // G, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nG, 1, G * G)
                attn_mask = torch.zeros((nG, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        
            share_q = self.q(self.norm0(h1_x))
            local_v = self.v(self.norm0(h1_x))

            h1_x_p = self.local_attn(share_q, share_q, local_v, self.pos_local, group_size=(Gh, Gw))
            h1_x = h1_x + h1_x_p

            h0_token = h1_x.mean(dim=1)
            h0_token = h0_token.reshape(b, Hp * Wp // G**2, c)
            h0_token = self.h0_norm(h0_token)
        
            tmp_x = torch.cat([h1_x, share_q, h1_x_p], dim=0)
            tmp_x = tmp_x.reshape(3*b, Hp // G, Wp // G, G, G, c).permute(0, 3, 1, 4, 2, 5).contiguous() # B, Hp//G, G, Wp//G, G, C

        # remove padding
        tmp_x = tmp_x.reshape(3*b, Hp, Wp, c)
        if pad_r > 0 or pad_b > 0:
            tmp_x = tmp_x[:, :H, :W, :].contiguous()
        tmp_x = tmp_x.view(3*b, H * W, c).contiguous()
        x, share_q, h1_x_p = tmp_x.reshape(3, b, H * W, c)
        h1_x_p = self.norm1(h1_x_p)
        h1_x_p = h1_x_p.permute(0, 2, 1).reshape(b, c, H, W)
        h1_x_p = self.aux_q(h1_x_p)
        h1_x_p = h1_x_p.flatten(2).permute(0, 2, 1)

        q = share_q + h1_x_p
        k = self.k(h0_token)
        v = self.v(h0_token)
        x = x + self.drop_path(self.global_attn(q, k, v, self.pos_global, H, W, group_size=(H, W, Hp // Gh, Wp // Gw)))
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

@BACKBONES.register_module()
class HLGTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], use_checkpoint=True, h0_h1_fusion_type='attn', h0_att=False,
                 proj_dwconv=None, downsampling=None, h0_h1_method=None, crs_interval=[8, 4, 2, 1], pretrained=None,
                 dynamic_position_bias=False,
                 ):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.use_checkpoint=use_checkpoint

        #############################################################

        # self.h0_h1_method = 'mean' # 'mean', 'hada', 'poolt'
        self.h0_h1_method = h0_h1_method # 'mean' 'hada' 'conv'    # mean is max
        self.h0_pos = True
        self.h0_h1_fusion_type = h0_h1_fusion_type
        self.proj_dwconv = proj_dwconv
        self.dynamic_position_bias = dynamic_position_bias

        self.s0_conv = nn.ModuleList()
        for i in range(2):
            down_sample = True if i == 0 else False
            inp = in_chans if i == 0 else embed_dims[0]
            self.s0_conv.append(conv_3x3_bn(inp=inp, oup=embed_dims[0], down_sample=down_sample))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # window_size, _, _, _ = find_best_hier(img_size // 4, 2)
        window_size = 7
        self.block1 = nn.ModuleList()
        for i in range(depths[0]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            down_sample = True if i == 0 else False
            inp = embed_dims[0] if i == 0 else embed_dims[0]
            self.block1.append(HLGLayers(
                input_dim = inp,
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[0], h0_att=h0_att, transform_method=self.h0_h1_method, h0_h1_fusion_type=self.h0_h1_fusion_type,
                window_size=window_size, lsda_flag=lsda_flag, interval=crs_interval[0], dynamic_position_bias=dynamic_position_bias, down_sample=down_sample))

        cur += depths[0]
        # window_size, _, _, _ = find_best_hier(img_size // 8, 2)
        window_size = 7
        self.block2 = nn.ModuleList()
        for i in range(depths[1]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            down_sample = True if i == 0 else False
            inp = embed_dims[0] if i == 0 else embed_dims[1]
            self.block2.append(HLGLayers(
                input_dim=inp,
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[1], h0_att=h0_att, transform_method=self.h0_h1_method, h0_h1_fusion_type=self.h0_h1_fusion_type,
                window_size=window_size, lsda_flag=lsda_flag, interval=crs_interval[1], dynamic_position_bias=dynamic_position_bias, down_sample=down_sample))

        cur += depths[1]
        # window_size, _, _, _ = find_best_hier(img_size // 16, 2)
        window_size = 7
        self.block3 = nn.ModuleList()
        for i in range(depths[2]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            down_sample = True if i == 0 else False
            inp = embed_dims[1] if i == 0 else embed_dims[2]
            self.block3.append(HLGLayers(
                input_dim=inp,
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[2], h0_att=h0_att, transform_method=self.h0_h1_method, h0_h1_fusion_type=self.h0_h1_fusion_type, 
                window_size=window_size, lsda_flag=lsda_flag, interval=crs_interval[2], dynamic_position_bias=dynamic_position_bias, down_sample=down_sample))

        cur += depths[2]
        # window_size, _, _, _ = find_best_hier(img_size // 32, 2)
        window_size = 7
        self.block4 = nn.ModuleList()
        for i in range(depths[3]):
            lsda_flag = 0 if (i % 2 == 0) else 1
            down_sample = True if i == 0 else False
            inp = embed_dims[2] if i == 0 else embed_dims[3]
            self.block4.append(HLGLayers(
                input_dim=inp,
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[3], h0_att=h0_att, transform_method=None, h0_h1_fusion_type=self.h0_h1_fusion_type, 
                window_size=window_size, lsda_flag=lsda_flag, interval=crs_interval[3], dynamic_position_bias=dynamic_position_bias, down_sample=down_sample))
        
        self.norm = norm_layer(embed_dims[3])

        # add a norm layer for each output
        self.norm_out_0 = norm_layer(embed_dims[0])
        self.norm_out_1 = norm_layer(embed_dims[1])
        self.norm_out_2 = norm_layer(embed_dims[2])

        self.img_size = img_size
        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            print("In init_weights")
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    
    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2).contiguous(),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1).contiguous()

    def hier_shape_spec(self, h, w):
        """"""
        # uhk, uhs, newH, uhp = find_best_hier(h, 2)
        # uwk, uws, newW, uwp = find_best_hier(w, 2)
        uhk, uhs, newH, uhp = 7, 7, h // 7, 0
        uwk, uws, newW, uwp = 7, 7, w // 7, 0
        h_config, w_config = (uhk, uhs, uhp, newH), (uwk, uws, uwp, newW)
        return h_config, w_config

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        
        for blk in self.s0_conv:
            x = blk(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        h_config, w_config = self.hier_shape_spec(H//2, W//2)

        for i, blk in enumerate(self.block1):
            if i == 1:
                H = H // 2
                W = W // 2
            inputs = (x, H, W, h_config, w_config)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, inputs)
            else:
                x = blk(inputs)
        x=self.norm_out_0(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        # stage 2
        h_config, w_config = self.hier_shape_spec(H//2, W//2)
        for i, blk in enumerate(self.block2):
            if i == 1:
                H = H // 2
                W = W // 2
            inputs = (x, H, W, h_config, w_config)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, inputs)
            else:
                x = blk(inputs)
        x=self.norm_out_1(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        # stage 3
        h_config, w_config = self.hier_shape_spec(H//2, W//2)
        for i, blk in enumerate(self.block3):
            if i == 1:
                H = H // 2
                W = W // 2
            inputs = (x, H, W, h_config, w_config)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, inputs)
            else:
                x = blk(inputs)
        x=self.norm_out_2(x)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())

        # stage 4
        h_config, w_config = self.hier_shape_spec(H//2, W//2)
        for i, blk in enumerate(self.block4):
            if i == 1:
                H = H // 2
                W = W // 2
            inputs = (x, H, W, h_config, w_config)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, inputs)
            else:
                x = blk(inputs)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return tuple(x)
