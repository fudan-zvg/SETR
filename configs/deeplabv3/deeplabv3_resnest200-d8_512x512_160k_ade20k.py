_base_ = './deeplabv3_r50-d8_512x512_160k_ade20k.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

_base_ = [
    '../_base_/models/deeplabv3_resnest200-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(depth=200),
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))
test_cfg = dict(mode='whole')
