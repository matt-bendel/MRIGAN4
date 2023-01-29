import os
import torch

import pytorch_lightning as pl

from models.GAN import GAN

torch.set_float32_matmul_precision('medium')
# init model
model = GAN()

# fit trainer on 128 GPUs
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp")
trainer.fit(model)
