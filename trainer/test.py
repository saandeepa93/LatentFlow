import torch 
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torchvision import utils
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('.')

import os
import random
import numpy as np
from sys import exit as e
from tqdm import tqdm
from math import log, sqrt, pi, cos
from icecream import ic
import argparse
import time
import matplotlib.pyplot as plt

import pickle
from sklearn import datasets

from models import LatentModel
from utils import plot_umap, gaussian_log_p, gaussian_sample, get_args, seed_everything, get_metrics
from configs import get_cfg_defaults
from loaders import CustomDataset, AffectDataset, RafDb
from losses import FlowConLoss
from trainer.robust_optimization import RobustOptimizer

def adjust_learning_rate(cfg, optimizer, epoch):
    lr = cfg.TRAINING.LR
    eta_min = lr * (cfg.LR.DECAY_RATE ** 3)
    lr = eta_min + (lr - eta_min) * (1 + cos(pi * epoch / cfg.TRAINING.ITER)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(cfg, epoch, batch_id, total_batches, optimizer):
  # T
  warm_epochs= cfg.LR.WARM_ITER
  warmup_from = cfg.LR.WARMUP_FROM
  warmup_to = cfg.TRAINING.LR
  if cfg.LR.WARM and epoch <= warm_epochs:
    p = (batch_id + (epoch - 1) * total_batches) / \
        (warm_epochs * total_batches)
    lr = warmup_from + p * (warmup_to - warmup_from)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
      #RAF_10
      # weight[2] = weight[2] * 2
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

def train(loader, epoch, model, optimizer, criterion, cfg, n_bins, device):
  avg_loss = []
  y_pred = []
  y_true = []
  model.train()
  for b, (image, exp, _) in enumerate(loader,0):
    if cfg.DATASET.AUG2:
      image = torch.cat(image, dim=0)
      exp = exp.repeat(2)
    
    image = image.to(device)
    exp = exp.to(device)

    # WARMUP LEARNING RATE
    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)

    # FORWARD 
    z = model.backbone(image).squeeze()
    out = model.fc(z)
    loss = criterion(out, exp)

    with torch.no_grad():
      avg_loss.append(loss.item())
      y_pred += torch.argmax(out, dim=-1).cpu().tolist()
      y_true += exp.cpu().tolist()  

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  avg_loss = sum(avg_loss)/len(avg_loss)
  return avg_loss,  y_pred, y_true

def validate(loader, model, criterion, cfg, n_bins, device):
  avg_loss = []
  y_pred = []
  y_true = []
  model.eval()
  for b, (image, exp, _) in enumerate(loader,0):
      
    image = image.to(device)
    exp = exp.to(device)

   # FORWARD 
    z = model.backbone(image).squeeze()
    out = model.fc(z)
    loss = criterion(out, exp)
    avg_loss.append(loss.item())
    y_pred += torch.argmax(out, dim=-1).cpu().tolist()
    y_true += exp.cpu().tolist()  

  avg_loss = sum(avg_loss)/len(avg_loss)

  return avg_loss,  y_pred, y_true

if __name__ == "__main__":
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

  # SET TENSORBOARD PATH
  # writer = SummaryWriter(f'./runs/{args.config}')

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)
  

  model = LatentModel(cfg)
  model.fc = nn.Linear(cfg.FLOW.IN_FEAT, cfg.DATASET.N_CLASS)
  model = model.to(device)

  print("number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  # PREPARE LOADER
  train_loader, val_loader = prepare_dataset(cfg)
  
  # PREPARE OPTIMIZER
  optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  scheduler = CosineAnnealingLR(optimizer, cfg.LR.T_MAX, cfg.LR.MIN_LR)

  criterion = nn.CrossEntropyLoss()

  # START TRAINING
  min_loss = 1e5
  n_bins = 0
  best_acc = 0
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for i in pbar:

    # TRAINING
    train_loss, y_pred_train, y_true_train = train(train_loader, i, model, optimizer, criterion, cfg, n_bins, device)
    with torch.no_grad():
      val_loss, y_pred_val, y_true_val = validate(val_loader, model, criterion, cfg, n_bins, device)
    
    train_acc, train_mcc, train_conf = get_metrics(y_pred_train, y_true_train)
    val_acc, val_mcc, val_conf = get_metrics(y_pred_val, y_true_val)

    # ADJUST LR
    if cfg.LR.ADJUST:
      scheduler.step()
      # if i > 25:
      #   scheduler.step()
        # adjust_learning_rate(cfg, optimizer, epoch)

    curr_lr = optimizer.param_groups[0]["lr"] 

    
    # BEST MODEL
    if val_acc > best_acc:
      min_loss = val_loss
      best_acc = val_acc
      ic(val_conf)

    # SAVE MODEL EVERY k EPOCHS
    # if i % 200 == 0:
    #   torch.save(model.state_dict(), f"checkpoint/{args.config}_model_{str(i + 1).zfill(6)}.pt")
      # torch.save(optimizer.state_dict(), f"checkpoint/{args.config}_optim_{str(i + 1).zfill(6)}.pt")

    
    pbar.set_description(
        f"train_loss: {round(train_loss, 4)}; train_acc: {round(train_acc, 4)};"
        f"val_loss: {round(val_loss, 4)}; val_acc: {round(val_acc, 4)};"
        f"best_acc: {round(best_acc, 4)};"
                        ) 
    
