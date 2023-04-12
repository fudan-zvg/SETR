# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='HLGTransformer',
        img_size=768,
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        depths=[2, 2, 6, 2],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        h0_att=False,
        proj_dwconv='convbn',
        downsampling=['c', 'sc', 'sc', 'sc'],
        h0_h1_method='mean',
        crs_interval=[8, 4, 2, 1],
        # pretrained=kwargs['pretrained'],
        dynamic_position_bias=True,
        in_chans=3,
        num_classes=19),
    decode_head=dict(
        type='HLGUPHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        fuse_level=0,
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
    # model training and testing settings
train_cfg=dict()
test_cfg=dict(mode='whole')
