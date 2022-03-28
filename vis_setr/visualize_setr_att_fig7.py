import torch
import torch.nn.functional as F
from mmseg.models.backbones.vit import VisionTransformer
import matplotlib.pyplot as plt
import mmcv
import os
import sklearn.decomposition as dec
import numpy as np
import cv2
import argparse

def draw_features(img,savename, cmap):
    fig = plt.figure()
    # plt.axis('off')
    pmin = np.min(img)
    pmax = np.max(img)
    img = (img - pmin) / (pmax - pmin + 0.000001)
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    # plt.imshow(img, cmap='gray')
    fig.savefig(savename)
    fig.clf()
    print(savename)

def draw_attention(img, attention_pt, savename, cmap):
    fig = plt.figure()
    plt.axis('off')
    pmin = np.min(img)
    pmax = np.max(img)
    img = (img - pmin) / (pmax - pmin + 0.000001)
    plt.imshow(img, cmap=cmap)
    plt.plot(attention_pt[1] * 16 + 15, attention_pt[0] * 16 + 15, color='r', marker='s', markersize=12)
    # plt.colorbar()
    # plt.imshow(img, cmap='gray')
    fig.savefig(savename)
    fig.clf()
    print(savename)

def draw_attention_origin(attn, img, attention_pt, savename, cmap):
    fig = plt.figure()
    plt.axis('off')
    pmin = np.min(attn)
    pmax = np.max(attn)
    print(pmin, pmax)
    attn = (attn - pmin) / (pmax - pmin + 0.000001)
    # attn[attn < 0] = 0
    plt.imshow(img)
    plt.imshow(attn, cmap=cmap, alpha=1)
    # plt.plot(attention_pt[1] * 16 + 15, attention_pt[0] * 16 + 15, color='r', marker='s', markersize=12)
    # plt.colorbar()
    # plt.imshow(img, cmap='gray')
    fig.savefig(savename)
    fig.clf()
    print(savename)

def draw_img(img,attention_pt,savename):
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.plot(attention_pt[1]*16+15, attention_pt[0]*16+15, color='r', marker='s', markersize=12)
    fig.savefig(savename)
    fig.clf()
    print(savename)

def upsample(img, size=480):
    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img_u = F.interpolate(img_t, size=size, mode='bilinear', align_corners=True)
    img_u = img_u[0,0,:,:].numpy()
    return img_u


def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


parser = argparse.ArgumentParser(description='CorrFlow')
parser.add_argument('-p', '--pic', help='layer num')
parser.add_argument('-a', '--pt1', help='pt1')
parser.add_argument('-b', '--pt2', help='pt2')
parser.add_argument('-g', '--gpu', help='pca feature')
args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))
file_client_args=dict(backend='disk')
savepath = './visualize/pascal/att_ocnet/'
file_client = mmcv.FileClient(backend='disk')
pca = dec.PCA(1)
outputs = []
norm_cfg = dict(type='SyncBN', requires_grad=True)
vit_model = VisionTransformer(img_size=480,
        align_corners=True,
        pre_syncbn_relu=False,
        pos_embed_interp=True,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=19,
        drop_rate=0.1,
        norm_cfg=norm_cfg,
        load_finetune=False).cuda()
state = torch.load('./model_path/pascal_context_iter_80000.pth')
vit_model.load_state_dict(state,strict=False)
vit_model.eval()
# attention_pt_sequence = [(14,14),(6,6),(6,23),(23,6),(23,23)]
# attention_pt_sequence = []
# for i in range(30):
#     for j in range(30):
#         attention_pt_sequence.append((i,j))
attention_pt_sequence = [(int(args.pt1),int(args.pt2))]
attns = []
filelist = os.listdir('./data/VOCdevkit/VOC2012/JPEGImages')
filename = filelist[int(args.pic)]
print(filename)
img = mmcv.imread('./data/VOCdevkit/VOC2012/JPEGImages/' + filename)
img = mmcv.bgr2rgb(img)
img_resize = cv2.resize(img, (480, 480))
img_print = img_resize
# draw_img(img_crop/255., savepath + "/berlin-{:03d}.png".format(count))
img_resize = mmcv.imnormalize(img_resize, np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]),True)
input = torch.tensor(img_resize)
input = input.permute(2, 0, 1).contiguous()
input = input.unsqueeze(0).cuda()
output, attn = vit_model(input)
# (1,16,2305,2305)
img_path = savepath + "{}-{}".format(args.pic, filename[:-4]) + '/'
if not os.path.exists(img_path):
    os.mkdir(img_path)
for attention_pt in attention_pt_sequence:
    img_att_path = img_path
    draw_img(img_print / 255., attention_pt,
             img_att_path + "origin-({:02d},{:02d}).png".format(attention_pt[0], attention_pt[1]))
    for head in range(16):
        attn_feature = attn[23][0, head, 1:, 1:].detach().cpu().numpy()
        attention_pt_flat = 30 * attention_pt[0] + attention_pt[1]
        map_attention_pt = attn_feature[attention_pt_flat, :].reshape((30, 30))
        draw_attention_origin(upsample(map_attention_pt, 480), img_print / 255., attention_pt,
                              img_att_path + "pic-{}-head-{:02d}-att-({:02d},{:02d}).png".format(int(args.pic), head, attention_pt[0], attention_pt[1]), 'Blues')

