import os
import glob

import numpy as np
from tqdm.notebook import tqdm
import albumentations as A

import torch
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from torch.utils.data import Dataset, DataLoader


SCAN_MAX = 400
SCAN_MIN = -1000

binarize_mask = lambda mask : (mask > 0).astype(np.float32)
min_max_scale_im = lambda x: ((x - SCAN_MIN) / (SCAN_MAX - SCAN_MIN)).astype(np.float32)
input_im_transform = lambda x: min_max_scale_im(np.clip(x, SCAN_MIN, SCAN_MAX))

class CADDataset(Dataset):
    def __init__(self, scan_dir, mask_dir, augment=False): #, info_file=None):
    
        print("***Dataset initialization can take several minutes!***")
        
        self.scan_files = sorted(glob.glob(os.path.join(scan_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
        self.scans = []
        self.masks = []
        
        scan_sizes = [0]
        for scan_file, mask_file in tqdm(zip(self.scan_files, self.mask_files), total=len(self.scan_files)):
            self.scans.append(np.load(scan_file))
            self.masks.append(np.load(mask_file))
            scan_sizes.append(self.scans[-1].shape[0])
        else:
            self.im_shape = self.scans[-1].shape[1:]

        self.scan_sizes_cumsum = np.cumsum(scan_sizes)
        
        self.augment = augment
        if self.augment:
            self.aug_transform = A.Compose([
                A.RandomScale(scale_limit=(0, 0.1), p=0.5),
                A.RandomCrop(*self.im_shape, always_apply=True, p=1),
                A.Rotate(limit=10, p=1),
            ])
            
        self.to_tensor = ToTensor()
        
        print("***Done!***")
        
    def __len__(self):
        return self.scan_sizes_cumsum[-1]
    
    def __getitem__(self, i):

        assert (i < len(self) and i >= 0), "Index out of range!"
        
        scan_idx = np.argwhere(i < self.scan_sizes_cumsum).min() - 1
        im_idx = i - self.scan_sizes_cumsum[scan_idx]
        
        im = self.scans[scan_idx][im_idx]
        mask = self.masks[scan_idx][im_idx]
        
        if self.augment:
            transformed = self.aug_transform(image=im, mask=mask)
            im = transformed['image']
            mask = transformed['mask']
        
        im = self.to_tensor(im)
        mask = self.to_tensor(mask).float()
        
        return im, mask
    
class CADDataloader(DataLoader):
    def __init__(self, dataset, batch_size=16, shuffle=False, num_workers=0):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        

def get_CADDataset(train_dir, val_dir, train=True, val=True, augment_data=False):
    outs = []
    if train:
        train_dataset = CADDataset(os.path.join(train_dir, 'images/'),
                                   os.path.join(train_dir, 'masks/'),
                                   augment=augment_data)
        outs.append(train_dataset)
    if val:
        val_dataset = CADDataset(os.path.join(val_dir, 'images/'),
                                 os.path.join(val_dir, 'masks/'),
                                 augment=False)
        outs.append(val_dataset)
    return outs