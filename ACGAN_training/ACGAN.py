import math
import pdb
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F


# from GlobalParameters import *
def onehot2label_tensor(onehot):
    return torch.argmax(onehot, dim=1)


def label2onehot_tensor(label, num_classes):
    return F.one_hot(label, num_classes)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_dim=3, label_dim=10, img_size=32, latent_dim=100):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(label_dim, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(128),
            nn.GroupNorm(8, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            # nn.BatchNorm2d(128, 0.8),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        return self.forward_with_features(noise, labels)[0]

    def forward_with_features(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        features = {"l1": out}
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        features["reshape"] = out
        for idx, layer in enumerate(self.conv_blocks):
            out = layer(out)
            features[f"conv_blocks.{idx}"] = out
        features["output"] = out
        return out, features

    @torch.no_grad()
    def get_intermediate_features(self, noise, labels, layer="reshape", detach=True):
        _, features = self.forward_with_features(noise, labels)
        if layer not in features:
            raise KeyError(f"Unknown generator layer '{layer}'. Available layers: {list(features.keys())}")
        feat = features[layer]
        return feat.detach() if detach else feat

    def available_feature_layers(self):
        names = ["l1", "reshape"] + [f"conv_blocks.{i}" for i in range(len(self.conv_blocks))] + ["output"]
        return names


class Discriminator(nn.Module):
    def __init__(self, img_dim=3, label_dim=10):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                # block.append(nn.BatchNorm2d(out_filters, 0.8))
                block.append(nn.GroupNorm(4, out_filters))
            return block

        self.blocks = nn.ModuleList([
            nn.Sequential(*discriminator_block(img_dim, 16, bn=False)),
            nn.Sequential(*discriminator_block(16, 32)),
            nn.Sequential(*discriminator_block(32, 64)),
            nn.Sequential(*discriminator_block(64, 128)),
        ])

        # The height and width of downsampled image
        # ds_size = size // 2 ** 4
        ds_size = 2

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, label_dim), nn.Softmax(dim=-1))

    def forward(self, img):
        validity, label, _ = self.forward_with_features(img)
        return validity, label

    def forward_with_features(self, img):
        out = img
        features = {"input": img}
        for idx, block in enumerate(self.blocks):
            out = block(out)
            features[f"blocks.{idx}"] = out
        out = out.view(out.shape[0], -1)
        features["flatten"] = out
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        features["adv_output"] = validity
        features["aux_output"] = label
        return validity, label, features

    @torch.no_grad()
    def get_intermediate_features(self, img, layer="flatten", detach=True):
        _, _, features = self.forward_with_features(img)
        if layer not in features:
            raise KeyError(f"Unknown discriminator layer '{layer}'. Available layers: {list(features.keys())}")
        feat = features[layer]
        return feat.detach() if detach else feat

    def available_feature_layers(self):
        names = ["input"] + [f"blocks.{i}" for i in range(len(self.blocks))] + ["flatten", "adv_output", "aux_output"]
        return names


class ACGAN(nn.Module):
    def __init__(self, img_dim=3, label_dim=10, img_size=32, latent_dim=100, device='cuda'):
        super(ACGAN, self).__init__()
        generator = Generator(img_dim, label_dim, img_size, latent_dim)
        discriminator = Discriminator(img_dim, label_dim)
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        self.generator, self.discriminator = generator, discriminator
        self.optim_g = torch.optim.Adam(generator.parameters(),  lr=0.0002, betas=(0.5, 0.999))
        self.optim_d = torch.optim.Adam(discriminator.parameters(),  lr=0.0002, betas=(0.5, 0.999))

        # self.criterion = nn.BCELoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()
        self.latent_dim = latent_dim
        self.n_classes = label_dim
        self.device = device

    def forward(self, noise, label):
        fake_imgs = self.generator(noise, label)
        return fake_imgs

    @torch.no_grad()
    def extract_features(self, network, x=None, labels=None, noise=None, layer="flatten", detach=True):
        network = network.lower()
        if network == "discriminator":
            if x is None:
                raise ValueError("x must be provided when network='discriminator'.")
            return self.discriminator.get_intermediate_features(x, layer=layer, detach=detach)
        if network == "generator":
            if noise is None:
                raise ValueError("noise must be provided when network='generator'.")
            if labels is None:
                raise ValueError("labels must be provided when network='generator'.")
            return self.generator.get_intermediate_features(noise, labels, layer=layer, detach=detach)
        raise ValueError("network must be either 'generator' or 'discriminator'.")

    def train_batch(self, imgs, labels):
        # x: samples [B C H W]
        # c: labels (one-hot) [B L]
        device = self.device
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, requires_grad=False, dtype=torch.float).to(device)
        fake = torch.zeros(batch_size, 1, requires_grad=False, dtype=torch.float).to(device)

        real_imgs, labels = imgs, onehot2label_tensor(labels)  # labels (not one-hot) [B 1]

        self.optim_g.zero_grad()
        # Sample noise and labels as generator input
        z = torch.from_numpy(np.random.normal(0, 1, (batch_size, self.latent_dim))).to(device).float()
        gen_labels = torch.from_numpy(np.random.randint(0, self.n_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = self.discriminator(gen_imgs)
        g_loss = 0.1 * self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels)

        g_loss.backward()
        self.optim_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optim_d.zero_grad()

        # Loss for real images
        real_pred, real_aux = self.discriminator(real_imgs)
        d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
        d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_corr_num = np.sum(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        self.optim_d.step()
        return d_loss.item(), g_loss.item(), d_corr_num, batch_size

    def sample_loop(self, sn, size):
        g_s_list, g_l_list = [], []
        n_once = 50
        for i in range(int(sn // n_once)):
            g_s, g_l = self.sample(n_once)
            g_s_list.append(g_s.to('cpu'))
            g_l_list.append(g_l.to('cpu'))
            if i % 5 == 0: print(f'sampling {i * n_once}/{sn}', end='\r')
        g_s_list = torch.cat(g_s_list, dim=0)
        g_l_list = torch.cat(g_l_list, dim=0)
        return g_s_list, g_l_list

    def sample(self, n_sample):
        device, n_classes = self.device, self.n_classes
        noise = torch.randn(n_sample, self.latent_dim).to(device)
        labels_one_hot = nn.functional.one_hot(torch.arange(n_classes), num_classes=n_classes)
        labels_one_hot = labels_one_hot.repeat((int(n_sample / labels_one_hot.shape[0]), 1)).to(device)
        labels = onehot2label_tensor(labels_one_hot)
        return self.generator(noise, labels), labels_one_hot

    # For FedSSL
    def fedssl_G_Reg(self, x, c, lbd):
        device = self.device
        images = x
        labels = onehot2label_tensor(c)  # labels (not one-hot) [B 1]
        labels_onehot = c  # onehot-labels [B num_classes], num_classes=10
        pvalidity, plabels = self.discriminator(images)
        real_loss = F.nll_loss(plabels, labels)


# if __name__ == "__main__":
#     device = 'cuda:0'
#     bs = 1
#     noise = torch.randn(bs, 100).to(device)
#     label = torch.randint(0, 10, (bs,)).to(device)
#     model = ACGAN().to(device)
#     # Calculate parameters
#     params = sum([param.nelement() for param in model.parameters()])
#     MACs = FlopCountAnalysis(model, (noise, label), ).total()
#     FLOPs = MACs * 2
#     print(f"Params: {params / 1e6} M, MACs: {MACs / 1e9} G, FLOPs: {FLOPs / 1e9} G")
#
#     print(f"Generator: {sum([param.nelement() for param in model.generator.parameters()]) / 1e6} M,"
#           f" Discriminator: {sum([param.nelement() for param in model.discriminator.parameters()]) / 1e6} M")
