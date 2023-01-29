import os

import torch
import torchvision
import pytorch_lightning as pl
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from models.architectures.generator import Generator
from models.architectures.discriminator import Discriminator

class GAN(pl.LightningModule):

    def __init__(self):
        super(GAN, self).__init__()
        # networks
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=512, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)

        # cache for generated images
        self.generated_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, _ = batch

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], 512).cuda()

            # generate images
            self.generated_imgs = self.forward(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)

            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            return d_loss

    def configure_optimizers(self):
        lr = 1e-3
        b1 = 0
        b2 = 0.99

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=32)