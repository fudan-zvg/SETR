import torch
import torch.nn.functional as F
from mmseg.models.backbones.vit import VisionTransformer
from mmseg.models.decode_heads.vit_fc_head import VisionTransformerFcHead
import matplotlib.pyplot as plt
import mmcv
import os
import sklearn.decomposition as dec
import numpy as np
import cv2
import argparse
import os

def draw_features(img,savename):
    fig = plt.figure()
    plt.axis('off')
    pmin = np.min(img)
    pmax = np.max(img)
    img = (img - pmin) / (pmax - pmin + 0.000001)
    plt.imshow(img, cmap='gray')
    # plt.imshow(img, cmap='gray')
    fig.savefig(savename)
    fig.clf()
    print(savename)

def draw_img(img,savename):
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img)
    fig.savefig(savename)
    fig.clf()
    print(savename)

def upsample(img, size=480):
    img_t = torch.tensor(img).permute(2,0,1).unsqueeze(0)
    img_u = F.interpolate(img_t, size=size, mode='bilinear', align_corners=True)
    img_u = img_u[0,:,:,:].permute(1,2,0).numpy()
    return img_u


parser = argparse.ArgumentParser(description='CorrFlow')
parser.add_argument('-l', '--layer', help='layer num')
parser.add_argument('-p', '--pca', help='pca feature')
parser.add_argument('-g', '--gpu', help='pca feature')
args = parser.parse_args()
torch.cuda.set_device(int(args.gpu))

file_client_args=dict(backend='disk')
savepath = './visualize/pascal/pup80000/upLayer/pca'+args.pca
if not os.path.exists(savepath):
    os.mkdir(savepath)
norm_cfg = dict(type='SyncBN', requires_grad=True)
vit_model = VisionTransformer(img_size=480,align_corners=False,pre_syncbn_relu=False,pos_embed_interp=True,drop_rate=0.,num_classes=60,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        norm_cfg=norm_cfg,
        load_finetune=False).cuda()
up_model = VisionTransformerFcHead(in_channels=1024,
                                   channels=512,
                                   in_index=23,embed_dim=1024,deconv_syncbn=False,
                                    loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                                   norm_cfg=norm_cfg,
                                   img_size=480,
                                   align_corners=False,
                                   num_fc=4,
                                   upsampling_method='bilinear',
                                   fc_syncbn=True,
                                   fc2conv=True,
                                   num_upsampe_layer=4,
                                   conv3x3_conv1x1=True,
                                   num_classes=60).cuda()
# state = torch.load('./pascal_context_iter_80000.pth')
state = torch.load('./model_path/pup_pascal_context_80k.pth')
vit_model.load_state_dict(state,strict=False)
vit_model.eval()
up_model.load_state_dict(state,strict=False)
up_model.eval()

for layer in [int(args.layer)]:
    outputs = []
    count = 0
    pca = dec.PCA(3)
    print("==============layer{}==============".format(str(layer+1)))
    width = 30
    for filename in os.listdir('./data/VOCdevkit/VOC2012/JPEGImages'):
        if count >= int(args.pca):
            break
        # img_bytes = file_client.get('./data/VOCdevkit/VOC2012/JPEGImages/' + filename)
        # img = mmcv.imfrombytes(
        #             img_bytes, flag='color', backend='cv2')
        # img = img.astype(np.float32)
        img = mmcv.imread('./data/VOCdevkit/VOC2012/JPEGImages/' + filename)
        img = mmcv.bgr2rgb(img)
        img_resize = cv2.resize(img, (480,480))
        # delta = (100, 500)
        # img_crop = img[0 + delta[0]:768 + delta[0], 0 + delta[1]:768 + delta[1], :]
        # draw_img(img_resize/255., savepath + "/origin/origin-{}-{}.jpg".format(str(count),filename[:-4]))
        img_crop = mmcv.imnormalize(img_resize, np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]),True)

        input = torch.tensor(img_crop).cuda()
        input = input.permute(2, 0, 1).contiguous()
        input = input.unsqueeze(0)
        mid,_ = vit_model(input)
        output = up_model(mid)
        width = output[layer].shape[-1]
        print(output[0].shape, output[1].shape, output[2].shape, output[3].shape,output[4].shape)
        # feature = output[23].detach().numpy()
        # img = feature[0,1:,:].T
        # print(img.shape)
        # img = np.reshape(img,(1024,48,48))

        print(output[layer].shape)
        feature = output[layer].view(1, -1, width * width).permute(0,2,1)[0]
        print(count,layer, 'feature shape:', feature.shape)
        feature = feature.detach().cpu().numpy()
        outputs.append(feature)
        # img = feature.view(-1,1024,48,48)
        # img = F.interpolate(img, size=768, mode='bilinear', align_corners=True)
        count += 1
    N = len(outputs)
    print("Total %d feature maps. Generating PCA images." % N)
    outputs = np.concatenate(outputs,0)
    print(outputs.shape)
    # np.save(savepath + '/outputs-{:02d}.npy'.format(layer), outputs)
    print('output shape: ',outputs.shape)
    outputs_pca = pca.fit_transform(outputs)
    print('pca shape: ',outputs_pca.shape)
    outputs_pca = (outputs_pca - outputs_pca.min()) / (outputs_pca.max() - outputs_pca.min())
    img_path = savepath + '/' + str(layer+1)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    for i in range(N):
        fig = plt.figure()
        plt.axis('off')
        image = outputs_pca[:width * width, :].reshape(width, width, 3)
        # os.makedirs(savepath + '/S{:02d}'.format(i), exist_ok=True)
        plt.imshow(upsample(image, 480))
        fig.savefig(img_path + "/pca-{:02d}-{:03d}.png".format(layer,i))
        fig.clf()
        # plt.imsave(savepath + "/F{:03d}.png".format(i), image)
        outputs_pca = outputs_pca[width * width:]