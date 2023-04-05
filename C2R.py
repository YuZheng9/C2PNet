import sys

import torch
import torch.nn as nn
from torchvision import models

sys.path.append('/')


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class C2R(nn.Module):
    def __init__(self, ablation=False):

        super(C2R, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation
        print('*******************use normal 6 neg clcr loss****************')

    def forward(self, a, p, n1, n2, n3, n4, n5, n6, inp, weight):
        a_vgg, p_vgg, n1_vgg, n2_vgg, n3_vgg, n4_vgg, n5_vgg, n6_vgg = self.vgg(a), self.vgg(p), self.vgg(n1), self.vgg(
            n2), self.vgg(n3), self.vgg(n4), self.vgg(n5), self.vgg(n6)
        inp_vgg = self.vgg(inp)
        n1_weight, n2_weight, n3_weight, n4_weight, n5_weight, n6_weight, inp_weight = weight
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an1 = self.l1(a_vgg[i], n1_vgg[i].detach())
                d_an2 = self.l1(a_vgg[i], n2_vgg[i].detach())
                d_an3 = self.l1(a_vgg[i], n3_vgg[i].detach())
                d_an4 = self.l1(a_vgg[i], n4_vgg[i].detach())
                d_an5 = self.l1(a_vgg[i], n5_vgg[i].detach())
                d_an6 = self.l1(a_vgg[i], n6_vgg[i].detach())
                d_inp = self.l1(a_vgg[i], inp_vgg[i].detach())
                contrastive = d_ap / (
                        d_an1 * n1_weight + d_an2 * n2_weight + d_an3 * n3_weight + d_an4 * n4_weight + d_an5 * n5_weight + d_an6 * n6_weight + d_inp * inp_weight + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss
