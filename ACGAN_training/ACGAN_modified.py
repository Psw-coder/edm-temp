import math
import pdb
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
# from fvcore.nn import FlopCountAnalysis, parameter_count
from ptflops import get_model_complexity_info
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import v2
import sys

sys.path.append('/home/ubuntu/Project/LX/DFLSemi/')


# from GlobalParameters import *
def g_lambda_function(lbd):
    return math.exp(max(lbd, 1 - lbd) - 1)


def onehot2label_tensor(onehot):
    return torch.argmax(onehot, dim=1)


def label2onehot_tensor(label, num_classes):
    return F.one_hot(label, num_classes)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self, img_dim=3, label_dim=10, img_size=32):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.padding = 1 if img_size == 32 else 3
        # input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        # input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    # nn.BatchNorm2d(256),
                                    nn.GroupNorm(8, 256),
                                    nn.ReLU(True))
        # input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    # nn.BatchNorm2d(128),
                                    nn.GroupNorm(8, 128),
                                    nn.ReLU(True))
        # input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, self.padding, bias=False),
                                    # nn.BatchNorm2d(64),
                                    nn.GroupNorm(8, 64),
                                    nn.ReLU(True))
        # input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, img_dim, 3, 1, 1, bias=False),
                                    nn.Tanh())
        # H_out = (H_in - 1) * s + d * (k - 1) - 2 * p + 1
        # if d = 1: H_out = (H_in - 1) * s + k - 2 * p

        # output 3*64*64

        self.embedding = nn.Embedding(label_dim, 100)

    def forward(self, noise, label):
        # noise: [B L] = [B 100]
        # label: [B 1] (not one-hot)
        if len(label.shape) == 2 and label.shape[-1] > 1: label = onehot2label_tensor(label)
        label_embedding = self.embedding(label)  # [B L] = [B 100]
        x = torch.mul(noise, label_embedding)
        x = x.view(-1, 100, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, img_dim=3, label_dim=10, img_size=32):
        super(Discriminator, self).__init__()
        self.img_dim = img_dim
        self.label_dim = label_dim
        self.img_size = img_size
        self.final_kernel_size = 4 if self.img_size == 32 else 3
        # input 3*32*32(3*28*28)
        self.layer1 = nn.Sequential(nn.Conv2d(img_dim, 64, 3, 1, 1, bias=False),
                                    # nn.BatchNorm2d(64),
                                    nn.GroupNorm(8, 64),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))

        # input 64*32*32(3*28*28)
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                    # nn.BatchNorm2d(128),
                                    nn.GroupNorm(8, 128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 128*16*16(3*14*14)
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                    # nn.BatchNorm2d(256),
                                    nn.GroupNorm(8, 256),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 256*8*8(3*7*7)
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                    # nn.BatchNorm2d(512),
                                    nn.GroupNorm(8, 512),
                                    nn.LeakyReLU(0.2, True))
        # H_out = ( H_in + 2 * p - d * (k - 1) - 1 ) / s + 1
        # if d = 1: H_out = (H_in + 2 * p - k) / s + 1

        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, self.final_kernel_size, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(512, label_dim + 1, self.final_kernel_size, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, self.label_dim + 1)
        return validity, plabel
        # return validity


class ACGAN(nn.Module):
    def __init__(self, img_dim=3, label_dim=10, img_size=32, device='cuda'):
        super(ACGAN, self).__init__()
        self.generator = Generator(img_dim, label_dim, img_size)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(img_dim, label_dim, img_size)

        # self.criterion = nn.BCELoss()
        self.device = device
        self.real_labels = 0.9 + 0.1 * torch.rand(10, device=device)
        self.fake_labels = 0.1 * torch.rand(10, device=device)

    def forward(self, noise, label):
        fake_imgs = self.generator(noise, label)
        return fake_imgs

    def calc_loss_dis(self, x, c, idx):
        # x: samples [B C H W]
        # c: labels (one-hot) [B L]
        device = self.device
        images = x
        labels = onehot2label_tensor(c)  # labels (not one-hot) [B 1]
        self.batch_size = images.size(0)
        real_label = self.real_labels[idx % 10]
        fake_label = self.fake_labels[idx % 10]
        fake_class_labels = 10 * torch.ones((self.batch_size,), dtype=torch.long, device=device)
        if idx % 25 == 0:
            real_label, fake_label = fake_label, real_label
        # real
        self.validity_label = torch.full((self.batch_size,), real_label, device=device)

        pvalidity, plabels = self.discriminator(images)
        # pvalidity = disc(images)

        errD_real_val = self.criterion(pvalidity, self.validity_label)
        errD_real_label = F.nll_loss(plabels, labels)

        errD_real = errD_real_val + errD_real_label
        errD_real.backward()

        # fake
        noise = torch.randn(self.batch_size, 100, device=device)
        sample_labels = torch.randint(0, 10, (self.batch_size,), device=device, dtype=torch.long)

        fakes = self.generator(noise, sample_labels)

        self.validity_label.fill_(fake_label)

        pvalidity, plabels = self.discriminator(fakes.detach())

        errD_fake_val = self.criterion(pvalidity, self.validity_label)
        errD_fake_label = F.nll_loss(plabels, fake_class_labels)

        errD_fake = errD_fake_val + errD_fake_label
        errD_fake.backward()
        return errD_real, errD_fake

    def sample_loop(self, sn, size):
        device = self.device
        g_s_list, g_l_list = [], []
        n_once = 50
        for i in range(int(sn // n_once)):
            # g_s, g_l = self.sample(n_once, size)
            g_s, g_l = self.sample(n_once, size)
            g_s_list.append(g_s.to('cpu'))
            g_l_list.append(g_l.to('cpu'))
            if i % 5 == 0: print(f'sampling {i * n_once}/{sn}', end='\r')
        g_s_list = torch.cat(g_s_list, dim=0)
        g_l_list = torch.cat(g_l_list, dim=0)
        return g_s_list, g_l_list

    def sample(self, n_sample, size):
        device = self.device
        label_dim_n = 10
        noise = torch.randn(n_sample, 100).to(device)
        labels = torch.randint(0, label_dim_n, (n_sample,)).to(device)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=label_dim_n).float()
        return self.generator(noise, labels), labels_onehot

    def seq_sample(self, size):
        device = self.device
        label_dim_n = 10
        n_sample = label_dim_n
        labels = torch.arange(label_dim_n).to(device)
        noise = torch.randn(n_sample, 100).to(device)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=label_dim_n).float()
        return self.generator(noise, labels), labels_onehot

    # For FedSSL
    def fedssl_G_Reg(self, x, c, lbd):
        device = self.device
        images = x
        labels = onehot2label_tensor(c)  # labels (not one-hot) [B 1]
        labels_onehot = c  # onehot-labels [B num_classes], num_classes=10
        pvalidity, plabels = self.discriminator(images)
        real_loss = F.nll_loss(plabels, labels)


def prepare_input(resolution):
    n = torch.randn(64, 100).to(device)
    lbl = torch.randint(0, 10, (64,)).to(device)
    return dict(noise=n, label=lbl)


if __name__ == "__main__":
    print(g_lambda_function(0.5))
    device = 'cuda:0'
    batch_size = 1
    noise = torch.randn(batch_size, 100).to(device)
    label = torch.randint(0, 10, (batch_size,)).to(device)
    model = ACGAN().to(device)
    # Calculate parameters
    params = sum([param.nelement() for param in model.parameters()])
    # MACs = FlopCountAnalysis(model, (noise, label), ).total()
    # FLOPs = MACs * 2
    # print(f"Params: {params / 1e6} M, MACs: {MACs / 1e9} G, FLOPs: {FLOPs / 1e9} G")
    generator = model.generator
    macs, params = get_model_complexity_info(generator, input_res=(1,), input_constructor=prepare_input,
                                             as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    print(f"Generator: {sum([param.nelement() for param in model.generator.parameters()]) / 1e6} M,"
          f" Discriminator: {sum([param.nelement() for param in model.discriminator.parameters()]) / 1e6} M")
