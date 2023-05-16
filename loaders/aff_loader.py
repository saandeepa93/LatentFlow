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
from collections import Counter
from sys import exit as e
from icecream import ic

from loaders.utils import GaussianBlur, TwoCropTransform


class AffectDataset(Dataset):
  def __init__(self, cfg, mode):
    super().__init__()

    self.cfg = cfg
    self.mode = mode
    self.aug = cfg.DATASET.AUG
    self.aug2 = cfg.DATASET.AUG2
    
    if mode == "train":
      root_dir = os.path.join(cfg.PATHS.DATA_ROOT, "train_set")
    elif mode == "val":
      root_dir = os.path.join(cfg.PATHS.DATA_ROOT, "val_set")
    self.root_dir = root_dir

    # Private dicts
    self.label_dict = {"Neutral": 0, "Happiness":1, "Sadness":2, "Surprise":3, "Fear":4, "Disgust":5, "Anger":6, "Contempt": 7}
    self.label_dict_inverse = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
    self.cnt_dict = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
    self.allowed_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    self.all_files_dict = self.__getAllFiles__()

    self.all_files = list(self.all_files_dict.keys())
    self.alb_train, self.alb_val = self.get_augmentation()
    self.all_labels = list(self.all_files_dict.values())
    ic(Counter(self.all_labels))
  

  def __getAllFiles__(self):
    all_files = {}
    nav_dir = os.path.join(self.root_dir, "images")
    for entry in os.scandir(nav_dir):
      fname_w = entry.name
      fpath = entry.path
      if os.path.splitext(fname_w)[-1] != ".jpg":
        continue
      fname = fname_w.split(".")[0]
      exp = int(np.load(os.path.join(self.root_dir, "annotations", fname+"_exp.npy")).item())
      if exp not in self.allowed_labels:
        continue
      if self.cnt_dict[exp] >= int(self.cfg.DATASET.COUNT):
        continue
        
      all_files[fpath] = exp
      self.cnt_dict[exp] += 1
    return all_files

  def get_augmentation(self):
    train_transform_dict = {}
    trans_probs = [0.5, 1., 0.7, 0.2, 0.2, 0.7, 0.5, 0.7]
    for i in range(self.cfg.DATASET.N_CLASS):
      train_transform_dict[i] = A.Compose([
        A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
        A.HorizontalFlip(p=trans_probs[i]),
        A.GaussianBlur(p=trans_probs[i]),
        A.Perspective(p=trans_probs[i]),
        A.Rotate(p=trans_probs[i]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ], 
        p=1)

    val_transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
      ToTensorV2()
      ], 
      p=1)
    return train_transform_dict, val_transforms

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    fpath = self.all_files[idx]
    fname = fpath.split("/")[-1].split(".")[0]

    image = np.array(Image.open(self.all_files[idx]))
    exp = self.all_files_dict[fpath]
    
    if self.mode== "train":
      if self.aug:
        transform = self.alb_train[exp](image=image)
      else:
        transform = self.alb_val(image=image)
    elif self.mode == "val":
      transform = self.alb_val(image=image)
    
    image_aug = transform['image']
    
    return image_aug, exp, fname