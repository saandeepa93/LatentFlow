import torch 
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2

import os
import re
import pandas as pd
import numpy as np
import glob
from sys import exit as e
from icecream import ic

from loaders.utils import GaussianBlur, TwoCropTransform


class RafDb(Dataset):
  def __init__(self, cfg, mode) -> None:
    super().__init__()

    # CLASS VARs
    self.cfg = cfg
    self.mode = mode
    self.aug = cfg.DATASET.AUG
    self.aug2 = cfg.DATASET.AUG2

    # PATHS and DFs
    self.data_dir = os.path.join(cfg.PATHS.DATA_ROOT, "Image", "aligned")
    label_dir = os.path.join(cfg.PATHS.DATA_ROOT, "EmoLabel", "list_patition_label.txt")
    self.labeldf = pd.read_csv(label_dir, sep=" ", header=None)
    self.labeldf.columns = ["fname", "exp"]
    self.labeldf['exp'] = self.labeldf['exp'] - 1

    # PRIVATE DICTS
    self.label_dict = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}
    self.cnt_dict = {"Surprise":0, "Fear":0, "Disgust":0, "Happiness":0, "Sadness":0, "Anger":0, "Neutral": 0}
    self.allowed_labels = [0, 1, 2, 3, 4, 5, 6]
    if len(self.allowed_labels) != cfg.DATASET.N_CLASS:
      raise ValueError("`N_CLASS` is different from `allowed_labels`")
    
    self.all_files_dict = self.getAllFiles()
    self.all_files = list(self.all_files_dict.keys())
    self.all_labels = list(self.all_files_dict.values())

    # AUGs
    self.alb_train, self.alb_val, self.aug_transforms = self.get_augmentation()
    print(self.cnt_dict)


  # def get_augmentation(self):
  #   train_transforms = A.Compose([
  #     A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
  #     A.HorizontalFlip(p=0.5),
  #     A.GaussianBlur(p=0.5),
  #     A.Rotate(p=0.5),
  #     A.RandomResizedCrop(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, scale=(0.5, 1.), p=0.5),
  #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
  #     ToTensorV2()
  #     ], 
  #     p=1)
  #   val_transforms = A.Compose([
  #     A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
  #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
  #     ToTensorV2()
  #     ], 
  #     # keypoints_params = A.KeypointParams(format='xy'), 
  #     p=1)
  #   return train_transforms, val_transforms, None
  
  def get_augmentation(self):
    train_transform_dict = {}
    trans_probs = [0.5, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5]
    for i in range(self.cfg.DATASET.N_CLASS):
      train_transform_dict[self.allowed_labels[i]] = A.Compose([
        A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
        A.HorizontalFlip(p=trans_probs[i]),
        A.GaussianBlur(p=trans_probs[i]),
        # A.RandomResizedCrop(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, scale=(0.5, 1.), p=trans_probs[i]),
        #UPDATED
        A.Perspective(p=trans_probs[i]),
        A.Rotate(p=trans_probs[i]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ], 
        p=1)

    train_transform_lst = transforms.Compose([
        transforms.RandomResizedCrop(self.cfg.DATASET.IMG_SIZE, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
    train_transform = TwoCropTransform(train_transform_lst)

    val_transforms = A.Compose([
      A.Resize(self.cfg.DATASET.IMG_SIZE, self.cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
      ToTensorV2()
      ], 
      p=1)
    return train_transform_dict, val_transforms, train_transform

  def getAllFiles(self):
    all_files_dict = {}
    for entry1 in os.scandir(self.data_dir):
      fpath = entry1.path
      fname_w_ext = entry1.name
      if os.path.splitext(fname_w_ext)[-1] not in ['.jpg', '.png']:
        continue
      
      fname_w_ext = re.sub("_aligned", "", fname_w_ext)
      fname = os.path.splitext(fname_w_ext)[0]
      fmode = fname.split('_')[0]

      expr = int(self.labeldf.loc[self.labeldf['fname']==fname_w_ext, 'exp'].item())
      label = self.label_dict[expr]
      
      # CONTINUE IF NO CONDITIONS BELOW MET
      if expr not in self.allowed_labels:
        continue
      if self.cnt_dict[label] >= int(self.cfg.DATASET.COUNT):
        continue
      if self.mode == "train" and fmode != "train":
        continue
      elif self.mode == "val" and fmode != "test":
        continue
      
      # APPEND FILEPATH AND LABEL
      all_files_dict[fpath] = expr
      self.cnt_dict[label] += 1

    return all_files_dict

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    fpath = self.all_files[idx]
    fname = os.path.splitext(fpath)[0].split('/')[-1]
    
    label = self.all_files_dict[fpath]

    image = Image.open(self.all_files[idx])
    image_arr = np.array(image).astype(np.float32)
    if self.mode== "train":
      if self.aug2:
        image_aug = self.aug_transforms(image)
      elif self.aug:
        transform = self.alb_train[int(label)](image=image_arr)
        # transform = self.alb_train(image=image_arr)
      elif not (self.aug or self.aug2):
        transform = self.alb_val(image=image_arr)
      image_aug = transform['image']
    else:
      transform = self.alb_val(image=image_arr)
      image_aug = transform['image']

    return image_aug, label, fname