import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from .hlg_utils import HLGHeadLayers


@HEADS.register_module()
class HLGUPHead(BaseDecodeHead):
    def __init__(self, fuse_level, depth, sr_ratio=4, **kwargs):
        super(HLGUPHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.fuse_level = fuse_level
        embed_dim = sum(self.in_channels)
        self.conv_0 = nn.Conv2d(embed_dim, self.channels, kernel_size=3, stride=1, padding=1)
        self.decode_transformer= nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1
            self.decode_transformer.append(HLGHeadLayers(
                self.channels, 
                self.channels//32, 
                sr_ratio=sr_ratio, 
                lsda_flag=lsda_flag, 
                interval=sr_ratio, 
                h0_att=False,
                dynamic_position_bias=True
        ))
        _, self.syncbn_fc_0 = build_norm_layer(self.norm_cfg, self.channels)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        fuse_shape = inputs[self.fuse_level].shape[2:]
        for lvl in range(len(inputs)):
            if inputs[lvl].shape[2:] != fuse_shape:
                inputs[lvl] = F.interpolate(
                        inputs[lvl], size=fuse_shape, mode='bilinear', align_corners=self.align_corners)
        x = torch.cat(inputs, dim=1)
        x = self.conv_0(x)
        x = self.syncbn_fc_0(x)
        x = F.relu(x, inplace=True)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        for blk in self.decode_transformer:
            inputs = (x, H, W, True, None, None, None, 1)
            x = blk(inputs)
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.cls_seg(x)
        x = F.interpolate(
            x, size=x.shape[-1]*4, mode='bilinear', align_corners=self.align_corners)
        
        return x
