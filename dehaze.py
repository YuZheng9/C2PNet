import argparse
import os

import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from metrics import psnr, ssim
from models.C2PNet import C2PNet

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name', help='name of dataset', choices=['indoor', 'outdoor'],
                    default='indoor')
parser.add_argument('--save_dir', type=str, default='dehaze_images', help='dehaze images save path')
parser.add_argument('--save', action='store_true', help='save dehaze images')
opt = parser.parse_args()

dataset = opt.dataset_name

if opt.save:
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    output_dir = os.path.join(opt.save_dir, dataset)
    print("pred_dir:", output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

if dataset == 'indoor':
    haze_dir = 'data/SOTS/indoor/hazy/'
    clear_dir = 'data/SOTS/indoor/clear/'
    model_dir = 'trained_models/ITS.pkl'
elif dataset == 'outdoor':
    haze_dir = 'data/SOTS/outdoor/hazy/'
    clear_dir = 'data/SOTS/outdoor/clear/'
    model_dir = 'trained_models/OTS.pkl'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = C2PNet(gps=3, blocks=19)
ckp = torch.load(model_dir)
net = net.to(device)
net.load_state_dict(ckp['model'])
net.eval()
psnr_list = []
ssim_list = []

for im in tqdm(os.listdir(haze_dir)):
    haze = Image.open(os.path.join(haze_dir, im)).convert('RGB')
    if dataset == 'indoor' or dataset == 'outdoor':
        clear_im = im.split('_')[0] + '.png'
    else:
        clear_im = im
    clear = Image.open(os.path.join(clear_dir, clear_im)).convert('RGB')
    haze1 = tfs.ToTensor()(haze)[None, ::]
    haze1 = haze1.to(device)
    clear_no = tfs.ToTensor()(clear)[None, ::]
    with torch.no_grad():
        pred = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    pp = psnr(pred.cpu(), clear_no)
    ss = ssim(pred.cpu(), clear_no)
    psnr_list.append(pp)
    ssim_list.append(ss)
    if opt.save:
        vutils.save_image(ts, os.path.join(output_dir, im))

print(f'Average PSNR is {np.mean(psnr_list)}')
print(f'Average SSIM is {np.mean(ssim_list)}')
