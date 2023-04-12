_base_ = [
    '../_base_/models/setr_pup_hlg.py',
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='HLGTransformer',
        pretrained='pretrain/pvt_large.pth',
        
        img_size=768,
        in_chans=3,
        patch_size=4,
        num_classes=19,
        
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        
        use_checkpoint=False,
        h0_att=False,
        proj_dwconv='convbn',
        downsampling=['c', 'sc', 'sc', 'sc'],
        h0_h1_method='mean',
        crs_interval=[8, 4, 2, 1],
        dynamic_position_bias=True,
        drop_path_rate=0.3
    ),
    decode_head=dict(
        type='HLGUPHead',
        depth=2,
        sr_ratio=8,
        in_channels=[128, 256, 512, 1024],
        num_classes=19,
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=19,
    ))
test_cfg = dict(mode='slide', crop_size=(768, 768), stride=(512, 512))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.0,
                 paramwise_cfg=dict(
                     custom_keys={
                         'norm': dict(decay_mult=0.),
                         'head': dict(lr_mult=10.),
                     })
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# crop_size = (768, 768)
find_unused_parameters = True
data = dict(samples_per_gpu=2)