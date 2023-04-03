import torch
import yaml
import os
import types
import json

import numpy as np
import sigpy as sp

from data_loaders.MRIDataModule import MRIDataModule
from datasets.fastmri_multicoil_general import FastMRIDataModule
from data_loaders.CelebAHQDataModule import CelebAHQDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.mri_unet import MRIUnet
from models.rcGAN import rcGAN
from models.CoModGAN import InpaintUNet
from utils.math import complex_abs, tensor_to_complex_np
from evaluation_scripts.metrics import psnr, ssim
from evaluation_scripts.fid.embeddings import VGG16Embedding
from evaluation_scripts.cfid.cfid_metric import CFIDMetric
from evaluation_scripts.fid.fid_metric import FIDMetric


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    args.mask_type = 1

    if args.default_model_descriptor:
        args.num_noise = 1

    if args.mri:
        with open('configs/mri/config.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = FastMRIDataModule(base_path=cfg.data_path, batch_size=cfg.batch_size)

        dm.setup()
        test_loader = dm.test_dataloader()
        if args.rcgan:
            model_alias = rcGAN
        else:
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

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    with torch.no_grad():
        model = model_alias.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/best_model.ckpt')
        model = model.cuda()
        model.eval()

        psnrs = []
        ssims = []
        apsds = []

        for i, data in enumerate(test_loader):
            y, x, mask, mean, std, maps, _, _ = data
            y = y.cuda()
            x = x.cuda()
            mask = mask.cuda()
            mean = mean.cuda()
            std = std.cuda()

            gens = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.in_chans / 2, cfg.im_size, cfg.im_size, 2)).cuda()
            for z in range(cfg.num_z_test):
                gens[:, z, :, :, :, :] = model.reformat(model.forward(y, mask))

            avg = torch.mean(gens, dim=1)

            gt = model.reformat(x)

            batch_psnrs = []
            batchs_ssims = []
            batch_apsds = []

            for j in range(y.size(0)):
                single_samps = np.zeros(cfg.num_z_test, cfg.im_size, cfg.im_size)

                S = sp.linop.Multiply((cfg.im_size, cfg.im_size), maps[j].cpu().numpy())
                gt_ksp, avg_ksp = tensor_to_complex_np((gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    (avg[j] * std[j] + mean[j]).cpu())

                avg_gen_np = torch.tensor(S.H * avg_ksp).abs().numpy()
                gt_np = torch.tensor(S.H * gt_ksp).abs().numpy()

                for z in range(cfg.num_z_test):
                    np_samp = tensor_to_complex_np((gens[j, z, :, :, :, :] * std[j] + mean[j]).cpu())
                    single_samps[z, :, :] = torch.tensor(S.H * np_samp).abs().numpy()

                apsds.append(np.mean(np.std(single_samps, axis=0), axis=(0, 1)))
                psnrs.append(psnr(gt_np, avg_gen_np))
                ssims.append(ssim(gt_np, avg_gen_np))

    inception_embedding = VGG16Embedding()

    cfids = []
    m_comps = []
    c_comps = []

    # CFID_1
    cfid_metric = CFIDMetric(gan=model,
                             loader=test_loader,
                             image_embedding=inception_embedding,
                             condition_embedding=inception_embedding,
                             cuda=True,
                             args=cfg,
                             ref_loader=False,
                             num_samps=1)

    cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    cfids.append(cfid)
    m_comps.append(m_comp)
    c_comps.append(c_comp)

    # CFID_2
    # cfid_metric = CFIDMetric(gan=model,
    #                          loader=val_dataloader,
    #                          image_embedding=inception_embedding,
    #                          condition_embedding=inception_embedding,
    #                          cuda=True,
    #                          args=cfg,
    #                          ref_loader=False,
    #                          num_samps=1)
    #
    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)

    # CFID_3
    # cfid_metric = CFIDMetric(gan=model,
    #                          loader=val_dataloader,
    #                          image_embedding=inception_embedding,
    #                          condition_embedding=inception_embedding,
    #                          cuda=True,
    #                          args=cfg,
    #                          ref_loader=train_dataloader,
    #                          num_samps=1)
    #
    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)

    # cfid_metric = FIDMetric(gan=model,
    #                         ref_loader=train_dataloader,
    #                         loader=test_loader,
    #                         image_embedding=inception_embedding,
    #                         condition_embedding=inception_embedding,
    #                         cuda=True,
    #                         args=cfg)
    #
    # cfid, m_comp, c_comp = cfid_metric.get_cfid_torch_pinv()
    # cfids.append(cfid)
    # m_comps.append(m_comp)
    # c_comps.append(c_comp)

    print(f'PSNR: {np.mean(psnrs)} \pm {np.std(psnrs) / np.sqrt(len(psnrs))}')
    print(f'SSIM: {np.mean(ssims)} \pm {np.std(ssims) / np.sqrt(len(ssims))}')
    print(f'APSD: {np.mean(apsds)}')
    for l in range(1):
        print(f'CFID_1: {cfids[l]:.2f}; M_COMP: {m_comps[l]:.4f}; C_COMP: {c_comps[l]:.4f}')
