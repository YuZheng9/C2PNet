import argparse,os
from tqdm import tqdm
from net.models.C2PDN import C2PDN
import torch
from PIL import Image
import torchvision.transforms as tfs
from net.metrics import psnr, ssim
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='its', help='its or ots or NH21 or NH19')
opt = parser.parse_args()
dataset = opt.task
if dataset == 'its':
    haze_dir = 'images/SOTS/indoor/hazy/'
    clear_dir = 'images/SOTS/indoor/clear/'
    model_dir = 'trained_models/ITS.pkl'
elif dataset == 'ots':
    haze_dir = 'images/SOTS/outdoor/hazy/'
    clear_dir = 'images/SOTS/outdoor/clear/'
    model_dir = 'trained_models/OTS.pkl'
elif dataset == 'NH19':
    haze_dir = 'images/NH19/hazy/'
    clear_dir = 'images/NH19/clear/'
    model_dir = 'trained_models/NH19.pkl'
elif dataset == 'NH21':
    haze_dir = 'images/NH21/hazy/'
    clear_dir = 'images/NH21/clear/'
    model_dir = 'trained_models/NH2.pkl'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


net = C2PDN(gps=3, blocks=19)
ckp = torch.load(model_dir)
net = net.to(device)
net.load_state_dict(ckp['model'])
net.eval()
psnr_list = []
ssim_list = []


for im in tqdm(os.listdir(haze_dir)):
    haze = Image.open(os.path.join(haze_dir,im)).convert('RGB')
    if dataset == 'its' or dataset == 'ots':
        clear_im = im.split('_')[0]+'.png'
    else:
        clear_im = im
    clear = Image.open(os.path.join(clear_dir,clear_im)).convert('RGB')
    haze1 = tfs.ToTensor()(haze)[None,::]
    haze1 = haze1.to(device)
    clear_no = tfs.ToTensor()(clear)[None,::]
    with torch.no_grad():
        # print(haze1.shape)
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    pp = psnr(pred.cpu(),clear_no)
    ss = ssim(pred.cpu(),clear_no)
    psnr_list.append(pp)
    ssim_list.append(ss)
    # im = im.split('.')[0]+'_' +str(pp) +'.png'
    # vutils.save_image(ts,output_dir+im)


print(np.mean(psnr_list))
print(np.mean(ssim_list))