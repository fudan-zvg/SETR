import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def draw_posemb(imgs, size, savename, cmap):
    fig, ax = plt.subplots(size, size, figsize=(10,10))
    for i in range(size):
        for j in range(size):
            id = i * size + j
            ax[i][j].pcolormesh(np.flipud(imgs[id]), cmap = cmap)
            ax[i][j].axis('off')
    fig.savefig(savename, format='png',  transparent=True, bbox_inches='tight', dpi=1000)
    fig.clf()
    print(savename)

parser = argparse.ArgumentParser(description='CorrFlow')
parser.add_argument('-p', '--path', help='path to model')
args = parser.parse_args()
savepath = './visualize/pascal/pos_embed'
# state = torch.load('./pascal_context_iter_80000.pth')
state = torch.load('model_path/jx_vit_large_p16_384-b3be5167.pth')
pos_embed = state['pos_embed']
# state = torch.load('model_path/' + args.path)
# state_dict = state['state_dict']
# pos_embed = state_dict['backbone.pos_embed']
# pos_embed = pos_embed[0,1:,:].numpy()
pos_embed = pos_embed[:,1:, :].permute(0,2,1).view(1,-1,24,24).contiguous()
pos_embed_up = torch.nn.functional.interpolate(pos_embed, (30,30),mode='bilinear', align_corners=True)
pos_embed_up = pos_embed_up.view(1,-1,900).contiguous()
pos_embed = pos_embed_up.permute(0,2,1).squeeze(0).numpy()

cs = cosine_similarity(pos_embed)
np.save('cos_sim_0.npy', cs)
# cs = np.load('cos_sim.npy')
cs_patch = []
for i in range(900):
    cs_patch.append(np.reshape(cs[i,:], (30,30)))

draw_posemb(cs_patch, 30, savepath + '/' + args.path[:-4] +'.png', 'viridis')