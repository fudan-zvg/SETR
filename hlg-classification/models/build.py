# --------------------------------------------------------
# Adopted from Swin Transformer
# --------------------------------------------------------

from .hlg import HLGTransformer


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'hlg':
        model = HLGTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.HLG.PATCH_SIZE,
            in_chans=config.MODEL.HLG.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.HLG.EMBED_DIMS,
            depths=config.MODEL.HLG.DEPTHS,
            sr_ratios=config.MODEL.HLG.SR_RATIOS,
            num_heads=config.MODEL.HLG.NUM_HEADS,
            mlp_ratios=config.MODEL.HLG.MLP_RATIO,
            qkv_bias=config.MODEL.HLG.QKV_BIAS,
            qk_scale=config.MODEL.HLG.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            h0_h1_method=config.MODEL.HLG.H0_H1_METHOD,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            dynamic_position_bias=config.MODEL.HLG.DPB
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
