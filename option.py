import argparse
import os
import warnings

import torch

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./data/')
parser.add_argument('--steps', type=int, default=1000000)
parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--resume', type=bool, default=True)
parser.add_argument('--eval_step', type=int, default=5000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir', type=str, default='./trained_models/')
parser.add_argument('--trainset', type=str, default='its_train')
parser.add_argument('--testset', type=str, default='its_test')
parser.add_argument('--net', type=str, default='C2PNet')
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--blocks', type=int, default=19, help='residual_blocks')
parser.add_argument('--bs', type=int, default=2, help='batch size')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int, default=240, help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--cl_lambda', type=float, default=0.25, help='cl_lambda')
parser.add_argument('--loss_weight', type=float, default=0.2, help='clcr_loss weight')
parser.add_argument('--clcrloss', action='store_true', help='clcr loss')
parser.add_argument('--norm', action='store_true', help='normalize')
parser.add_argument('--ablation_type', type=int, default=0, help='ablation_type')
parser.add_argument('--clip', action='store_true', help='use grad clip')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = opt.trainset + '_' + opt.net.split('.')[0] + '_' + str(opt.gps) + '_' + str(opt.blocks) + '_' + str(
    opt.name)

if opt.clcrloss:
    model_name = model_name + '_clcr'

opt.latest_model_dir = opt.model_dir + model_name + '_latest.pk'

opt.model_dir = opt.model_dir + model_name + '.pk'
log_dir = 'logs/' + model_name

print(opt)
print('model_dir:', opt.model_dir)

if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
    os.mkdir('numpy_files')
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('samples'):
    os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
    os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
