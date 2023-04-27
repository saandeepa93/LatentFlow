import torch 
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2

import os
import pandas as pd
import numpy as np
import glob
from sys import exit as e
from icecream import ic

from loaders.utils import GaussianBlur, TwoCropTransform

class CustomDataset(Dataset):
  def __init__(self, cfg, mode, aug=False):
    super().__init__()

    self.cfg = cfg
    self.mode = mode
    self.aug = aug
    self.all_files = glob.glob(os.path.join(cfg.PATHS.DATA_ROOT, "*", "*2D.bmp"))
    if mode == "train":
      self.all_files = [f for f in self.all_files if cfg.DATASET.TEST not in f]
    elif mode == "val":
      self.all_files = [f for f in self.all_files if cfg.DATASET.TEST in f]
    self.exp_dict = {"NE": 0, "AN": 1, "DI": 2, "FE": 3, "HA": 4, "SA": 5, "SU": 6}
    self.alb_train, self.alb_val = self.get_augmentation()

    self.gauss = GaussianBlur(kernel_size = int(0.1 * cfg.DATASET.IMG_SIZE))

  

  def get_augmentation(self):
    train_transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.HorizontalFlip(p=0.5),
      # A.GaussianBlur(p=0.5),
      # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
      ToTensorV2()
      ], 
      p=1)
    val_transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
      ToTensorV2()
      ], 
      # keypoints_params = A.KeypointParams(format='xy'), 
      p=1)
    return train_transforms, val_transforms

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    # paths
    fpath = self.all_files[idx]
    fname = os.path.splitext(fpath)[0].split("/")[-1]

    # Load and augment image
    # try:
    image = np.array(Image.open(fpath)).astype(np.float32)
    if self.mode== "train":
      transform = self.alb_train(image=image)
    elif self.mode == "val":
      transform = self.alb_val(image=image)
    image = transform['image']
    if self.aug:
      image_aug = [self.gauss(image) for _ in range(self.cfg.DATASET.N_VIEWS)]
    else:
      image_aug = image
    # Load labels
    exp = self.exp_dict[fname.split("_")[1][:2]]
    return image_aug, exp, fname