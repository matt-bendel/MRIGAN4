import torch
import yaml
import os
import types
import json

import numpy as np

from data_loaders.MRIDataModule import MRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from evaluation_scripts.fid.embeddings import VGG16Embedding
from models.mri_unet import MRIUnet
from models.inpaint_unet import InpaintUNet

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    if args.default_model_descriptor:
        args.num_noise = 1

    if args.mri:
        with open('configs/mri/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MRIDataModule(cfg, args.mask_type)
        dm.setup()
        test_loader = dm.test_dataloader()
        model_alias = MRIUnet
    elif args.inpaint:
        with open('configs/inpaint/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = CelebAHQDataModule(cfg, args.mask_type)
        dm.setup()
        test_loader = dm.test_dataloader()
        model_alias = InpaintUNet
    elif args.cs:
        args.checkpoint_dir = "/storage/matt_models/cs_baseline/"
        # TODO: LSUN?? data module
        # TODO: CS Lighnting Module
        pass
    else:
        print("No valid application selected. Please include one of the following args: --mri, --inpaint, --cs.")
        exit()

    inception_embedding = VGG16Embedding()

    with torch.no_grad():
        # model = model_alias.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/best_model.ckpt')
        model = model_alias.load_from_checkpoint(checkpoint_path=cfg.checkpoint_dir + 'genie/best_model.ckpt')
        model = model.cuda()
        model.eval()

        psnrs = []
        ssims = []

        for i, data in enumerate(test_loader):
            losses = model.external_test_step(data, i)
            psnrs.append(losses['psnr'])
            ssims.append(losses['ssim'])

    print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
