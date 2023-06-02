import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import utils
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.distributions.normal import Normal

import os
import random
import numpy as np
from sys import exit as e
from tqdm import tqdm
from math import log, sqrt, pi
from icecream import ic
import argparse
import time
import sys
sys.path.append('.')

from models import LatentModel
from loaders import CustomDataset, AffectDataset, RafDb
from losses import FlowConLoss
from configs import get_cfg_defaults
import utils as ut


allowed_labels = [0, 1, 2, 3, 4, 5, 6]


def prepare_dataset(cfg):
  # PREPARE LOADER
  if cfg.DATASET.DS_NAME == "BU3D":
    train_dataset = CustomDataset(cfg, "train")
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
    val_dataset = CustomDataset(cfg, "val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
  
  elif cfg.DATASET.DS_NAME == "AFF":
    train_dataset = AffectDataset(cfg, "train")
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
    val_dataset = AffectDataset(cfg, "val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
    
  elif cfg.DATASET.DS_NAME == "RAF":
    train_dataset = RafDb(cfg, "train")
    if cfg.DATASET.W_SAMPLER:
      class_freq = np.array(list(train_dataset.cnt_dict.values()))
      weight = 1./class_freq
      sample_weight = torch.tensor([weight[t] for t in train_dataset.all_labels])
      sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight))
      train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, \
        num_workers=cfg.DATASET.NUM_WORKERS, sampler=sampler)
    else:
      sampler=None
      train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
        num_workers=cfg.DATASET.NUM_WORKERS)
    
    val_dataset = RafDb(cfg, "val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
    
  return train_loader, val_loader

def save_flow_embedding(cfg, args, loader, model, mode, device):
  n_bins = 2.0 ** cfg.FLOW.N_BITS
  with torch.no_grad():
    all_fname = []
    all_label = []
    
    z_dict, mu_dict, std_dict = [], [], []
    with torch.no_grad():

      # COLLECT FEATURES, MEAN AND LOG_SDs
      for b, (x, label, fname) in enumerate(tqdm(loader), 0):
        x = x.to(device)
        label = label.to(device)

        features, means, log_sds, logdet, *_   = model(x)
        z_dict.append(features)
        mu_dict.append(means)
        std_dict.append(log_sds)

        all_fname += list(fname)
        all_label.append(label)

      # STACK INTO TENSORS
      all_label = torch.cat(all_label, dim=0)
      z = torch.cat(z_dict, dim=0)

      # PLOT UMAPS
      ut.plot_umap(cfg, z.cpu(), all_label.cpu(), f"{args.config}", all_fname, 2, mode)
      ut.plot_umap(cfg, z.cpu(), all_label.cpu(), f"{args.config}", all_fname, 3, mode)

      # SAVE MEANs and LOG_SDs
      if mode == "train":
        mu_k, std_k = [], []
        for cls in range(cfg.DATASET.N_CLASS):
          cls_indices = (all_label == cls).nonzero().squeeze()
          mu_k.append(torch.index_select(torch.cat(mu_dict, dim=0), 0, cls_indices).mean(0))
          std_k.append(torch.index_select(torch.cat(std_dict, dim=0), 0, cls_indices).mean(0))
        
        dist_dir = os.path.join("./data/distributions", f"{args.config}")
        ut.mkdir(dist_dir)
        torch.save(mu_k, os.path.join(dist_dir, "mu.pt"))
        torch.save(std_k, os.path.join(dist_dir, "log_sd.pt"))




if __name__ == "__main__":
  ut.seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.enabled = True
  print(torch.__version__) 
  print(torch.version.cuda) 
  print(torch.backends.cudnn.version()) #

  print("GPU: ", torch.cuda.is_available())

  args = ut.get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.DATASET.AUG = False
  cfg.DATASET.W_SAMPLER = False
  cfg.DATASET.NUM_WORKERS = 1
  cfg.TRAINING.BATCH = 128
  cfg.freeze()
  print(cfg)
  

  model = LatentModel(cfg)
  model = model.to(device)
  checkpoint = torch.load(f"./checkpoints/{args.config}_model_final.pt", map_location=device)
  model.load_state_dict(checkpoint)
  print("Total Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  model.eval()
  # NLL & CONTRASTIVE LOSS
  criterion = FlowConLoss(cfg, device)

  # PREPARE LOADER
  train_loader, val_loader = prepare_dataset(cfg)
  print("Saving Train dataset...")
  save_flow_embedding(cfg, args, train_loader, model, "train", device)
  print("Saving Val dataset...")
  save_flow_embedding(cfg, args, val_loader, model, "val", device)


