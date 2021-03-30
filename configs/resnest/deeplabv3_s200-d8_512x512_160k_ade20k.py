_base_ = '../deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k.py'
model = dict(
    pretrained='open-mmlab://resnest200',
    backbone=dict(
        type='ResNeSt',
        depth=200,
        stem_channels=128,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True))

