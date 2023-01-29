import os

import pytorch_lightning as pl

from models.GAN import GAN

# init model
model = GAN(None)

# fit trainer on 128 GPUs
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp")
trainer.fit(model)