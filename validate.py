import torch
import os

import numpy as np
import pytorch_lightning as pl

from data_loaders.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.in_chans = 16
    args.out_chans = 16

    args.checkpoint_dir = "/storage/matt_models"
    dm = MRIDataModule(args)
    dm.setup()
    val_loader = dm.val_dataloader()
    best_epoch = -1
    inception_embedding = VGG16Embedding()
    best_cfid = 10000000

    with torch.no_grad():
        for epoch in range(150, 200):
            print(f"VALIDATING EPOCH: {epoch + 1}")
            model = rcGAN.load_from_checkpoint(checkpoint_path=args.checkpoint_dir + f'/checkpoint-epoch={epoch}.ckpt')
            if model.is_good_model == 0:
                print("NO GOOD: SKIPPING...")
                continue

            model = model.cuda()
            model.eval()

            cfid_metric = CFIDMetric(gan=model,
                                     loader=val_loader,
                                     image_embedding=inception_embedding,
                                     condition_embedding=inception_embedding,
                                     cuda=True,
                                     args=args,
                                     ref_loader=False,
                                     num_samps=1)

            cfids = cfid_metric.get_cfid_torch_pinv()

            cfid_val = np.mean(cfids)

            if cfid_val < best_cfid:
                best_epoch = epoch
                best_cfid = cfid_val

    print(f"BEST EPOCH: {best_epoch}")

    for epoch in range(150, 200):
        if epoch != best_epoch:
            os.remove(args.checkpoint_dir + f'/checkpoint-epoch={epoch}.ckpt')
