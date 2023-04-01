#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff
"""
import pandas as pd
import requests
import pickle
import torch
import pytorch_lightning as pl
import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.get_mask import get_mask

from typing import Dict, Optional, Sequence, Tuple, Union

import sigpy as sp
import sigpy.mri as mr

import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np, normalize, normalize_instance
from datasets.fastmri_multicoil_preprocess import retrieve_metadata, et_query, fetch_dir, ImageCropandKspaceCompression


import time


def get_compressed(kspace: np.ndarray, img_size, num_vcoils = 8, vh = None):
    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []
    

    #Compress to 8 virtual coils and crop
    compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img).transpose(1,2,0),
                                                   img_size, num_vcoils, vh)

        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(to_tensor(compressed_img).permute(2,0,1,3))
    
    
    return compressed_img, compressed_k


class MulticoilTransform:
    
    def __init__(self, mask_type = None, img_size=320, accel_rate = 4, num_vcoils=8):

        self.mask_type = mask_type
        self.img_size = img_size
        self.accel_rate = accel_rate
        self.num_vcoils = num_vcoils

        self.mask = get_mask(self.img_size, R=8)
        
    def __call__(self, kspace, max_val, vh):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.
        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        #kspace is dimension [num_coils, size0, size1, 2]
        kspace = to_tensor(kspace)
        
        #Compress to virtual coils
        gt_img, gt_k = get_compressed(kspace, self.img_size, num_vcoils = self.num_vcoils, vh = vh)

        # Stack the coils and real and imaginary
        gt_img = to_tensor(gt_img).permute(2,3,0,1).reshape(-1,self.img_size,self.img_size).unsqueeze(0)

        #Apply the mask
        masked_kspace = gt_k * self.mask
        
        #Get the zf imgs
        masked_img = fastmri.ifft2c(masked_kspace).permute(0,3,1,2).reshape(-1,self.img_size,self.img_size).unsqueeze(0)

        #Normalized based on the 95th percentile max value of the magnitude
        zf_img, mean, std = normalize_instance(masked_img)
        gt_img = normalize(gt_img, mean, std)
        

        return zf_img, gt_img, self.mask, mean, std
    


class MulticoilDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_val_dir, img_size=320, mask_type = 's4', accel_rate = 4, scan_type = None, num_vcoils = 8, slice_range=None, val=False, test=False):
        ''' 
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK' for knee
        scan_type: None, 'AXT2'
        '''

        self.root = root
        self.img_size = img_size
        self.mask_type = mask_type
        self.accel_rate = accel_rate
        self.max_val_dir = max_val_dir
        self.examples = []
        self.val = val
        self.test = test
        
        self.multicoil_transf = MulticoilTransform(mask_type=self.mask_type,
                                                img_size = self.img_size,
                                                accel_rate = 8,
                                                num_vcoils = num_vcoils,
                                                )
        
        self.slice_range = slice_range
        

        files = list(Path(root).iterdir())

        print('Loading Data')
        for fname in tqdm(sorted(files)):
            
            #Skip non-data files
            if fname.name[0] == '.':
                continue
            
            #Recover the metadata
            metadata, num_slices = retrieve_metadata(fname)
            
            with h5py.File(fname, "r") as hf:
                
                #Get the attributes of the volume
                attrs = dict(hf.attrs)
                attrs.update(metadata)
                
                
                if scan_type is not None:
                    if attrs["acquisition"] != scan_type or attrs['encoding_size'][1] < img_size or hf['kspace'].shape[1] <= num_vcoils:
                        continue

                if attrs['encoding_size'][1] < img_size or hf['kspace'].shape[1] <= num_vcoils:
                    continue

                num_slices = hf['kspace'].shape[0]

                #Use all the slices if a range is not specified
                if self.slice_range is None:
                    #num_slices = hf['kspace'].shape[0]
                    slice_range = [0, num_slices]
                else:
                    if type(self.slice_range) is list:
                        if self.slice_range[1] > num_slices:
                            slice_range = [self.slice_range[0], num_slices]
                        elif self.slice_range[0] > num_slices:
                            raise Exception('Problem with {}, the lower range of slice range is > the number of slices'.format(fname.name[0]))
                        else:
                            slice_range = self.slice_range
                    elif self.slice_range < 1.0:
                        #num_slices = hf['kspace'].shape[0]
                        #Use percentage of center slices (i.e. center 80% of slices)
                        slice_range = [int(num_slices * (1-self.slice_range)), int(num_slices*self.slice_range)]

                
                
                
            self.examples += [(fname, slice_ind) for slice_ind in range(slice_range[0],slice_range[1])]


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]
        #print(i, fname, dataslice)
        
        with h5py.File(os.path.join(self.max_val_dir, fname.name), 'r') as hf:
            max_val = hf.attrs['max_val']

            vh = hf['vh'][dataslice]
            #vh = None
        
        
        with h5py.File(fname, "r") as hf:
            
            #Get the compressed target kspace
            kspace = hf['kspace'][dataslice]
            
            acquisition = hf.attrs['acquisition']
            
            zf_img, gt_img, mask, mean, std = self.multicoil_transf(kspace=kspace, max_val=max_val, vh=vh)
            
            zf_img = zf_img.squeeze(0)
            gt_img = gt_img.squeeze(0)

        maps = 0

        if self.val:
            with open(f'/storage/fastMRI_brain/sense_maps/val_full_res/{fname.name}_{dataslice}.pkl', 'rb') as inp:
                maps = pickle.load(inp)

        if self.test:
            with open(f'/storage/fastMRI_brain/sense_maps/test_full_res/{fname.name}_{dataslice}.pkl', 'rb') as inp:
                maps = pickle.load(inp)
            
        print(maps.shape)
        return (
            zf_img.float(),
            gt_img.float(),
            mask,
            mean,
            std,
            maps,
            fname.name,
            dataslice
        )



    
class FastMRIDataModule(pl.LightningDataModule):
    def __init__(self, base_path, batch_size:int = 32, num_data_loader_workers:int = 20, annotated=False,**kwargs):
        """
        Initialize the data module for the LoDoPaB-CT dataset.

        Parameters
        ----------
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 8.

        Returns
        -------
        None.

        """
        super().__init__()

        self.batch_size = batch_size
        self.num_data_loader_workers = num_data_loader_workers
        #self.data_range = [-6, 6]
        self.annotated = annotated
        
        self.base_path = base_path
        self.accel_rate = 8
        self.img_size = 384
        self.use_complex = True
        self.challenge = "multicoil"
        #self.mri_type = kwargs['mri_type'] #'knee', 'brain'
        self.mask_type = "gro"
        self.num_vcoils = 8
        
        self.slice_range = 8

        self.mri_type = 'brain'
        self.scan_type = "AXT2"
        self.specific_label = None

    def prepare_data(self):
        """
        Preparation steps like downloading etc. 
        Don't use self here!

        Returns
        -------
        None.

        """
        None

    def setup(self, stage:str = None):
        """
        This is called by every GPU. Self can be used in this context!

        Parameters
        ----------
        stage : str, optional
            Current stage, e.g. 'fit' or 'test'.
            The default is None.

        Returns
        -------
        None.

        """
        train_dir = os.path.join(self.base_path, '{0}_{1}'.format(self.challenge, 'train'))
        val_dir = os.path.join(self.base_path, '{0}_{1}'.format(self.challenge, 'val'))

        
        max_val_dir_train = os.path.join(self.base_path, '{0}_{1}_{2}coils_general'.format(self.challenge, 'train', self.num_vcoils))
        max_val_dir_val = os.path.join(self.base_path, '{0}_{1}_{2}coils_general'.format(self.challenge, 'val', self.num_vcoils))


        # Assign train/val datasets for use in dataloaders
        self.train = MulticoilDataset(train_dir,
                                      max_val_dir_train,
                                      self.img_size, self.mask_type,
                                      self.accel_rate, "AXT2",
                                      self.num_vcoils,
                                      [0, 8]
                                      )
        self.val = MulticoilDataset(val_dir,
                                    max_val_dir_val,
                                    self.img_size, self.mask_type,
                                    self.accel_rate, "AXT2",
                                    self.num_vcoils,
                                    [0, 8],
                                    val=True
                                    )

        if self.mri_type == 'brain':
            test_dir = os.path.join(self.base_path, 'small_T2_test')
            max_val_dir_test = os.path.join(self.base_path, 'multicoil_small_T2_test_8coils')
        
            self.test = MulticoilDataset(test_dir,
                                        max_val_dir_test,
                                        self.img_size, self.mask_type,
                                        self.accel_rate, "AXT2",
                                        self.num_vcoils,
                                        [0,6],
                                        test=True
                                        )



    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        return DataLoader(self.train, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=True, pin_memory=False, drop_last=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=False, drop_last=True)
    
    def test_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=False)


    
    
if __name__ == '__main__':
    mri_type = 'brain'
    annotation_version = None

    if mri_type == 'brain':
        base_dir = "/storage/fastMRI_brain/data/"
    else:
        base_dir = "/storage/fastMRI/data/"


    kwargs = {'center_frac': 0.0807,
              'accel_rate': 8,
              'img_size': 384,
              'complex': True,
              'challenge': 'multicoil',
              'vol': True,
              'img_type': 'mag',
              'mask_type': 'matt2',
              'scan_type': None,
              'num_vcoils': 8,
              'slice_range': [0,8]
              }

    data = FastMRIDataModule(base_dir, batch = 16, **kwargs)
    data.prepare_data()
    data.setup()

    for i in range(len(data.train)):

        print(i)
        x = data.train[i]

