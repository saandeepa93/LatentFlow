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
from collections import Counter
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
from utils import plot_umap, gaussian_log_p, gaussian_sample, get_args, seed_everything, plot_loader_imgs
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
    if cfg.DATASET.W_SAMPLER:
      class_freq = np.array(list(train_dataset.cnt_dict.values()))
      weight = 1./class_freq
      sample_weight = torch.tensor([weight[t] for t in train_dataset.all_labels])
      sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight))
      train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=False, \
        sampler=sampler, num_workers=cfg.DATASET.NUM_WORKERS)
    else:
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
  avg_con_loss = []
  avg_nll_loss = []
  robust = False
  model.train()
  for b, (image, exp, _) in enumerate(loader,0):
    image = image.to(device)
    exp = exp.to(device)

    # WARMUP LEARNING RATE
    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)

    # FORWARD 
    z, means, log_sds, sldj, log_vars = model(image)

    # LOSS
    nll_loss, log_p, _, log_p_all = criterion.nllLoss(z, sldj, means, log_sds)
    # con_loss = criterion.conLoss(log_p_all, exp)
    con_loss = criterion.kl_div(means, log_sds, exp, log_p_all)
    con_loss_mean = con_loss.mean()

    # loss = con_loss_mean + (cfg.TRAINING.LMBD * nll_loss)
    pre1 = torch.exp(-log_vars[0])
    pre2 = torch.exp(-log_vars[1])
    loss = ((pre1) * con_loss_mean) + ((pre2) * nll_loss) + (log_vars[0] + log_vars[1])
    # loss = nll_loss + con_loss_mean

    with torch.no_grad():
      avg_con_loss += con_loss.tolist()
      # avg_con_loss.append(con_loss_mean.item())
      avg_nll_loss.append(nll_loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  avg_con_loss = sum(avg_con_loss)/len(avg_con_loss)
  avg_nll_loss = sum(avg_nll_loss)/len(avg_nll_loss)
  # avg_con_loss = -1

  return avg_con_loss, avg_nll_loss, log_p.mean(), log_vars

def validate(loader, model, criterion, cfg, n_bins, device):
  total_con_loss = []
  total_nll_loss = []
  model.eval()
  for b, (image, exp, _) in enumerate(loader,0):
      
    image = image.to(device)
    exp = exp.to(device)

    # FORWARD 
    z, means, log_sds, sldj, log_vars = model(image)

    nll_loss, log_p, _, log_p_all = criterion.nllLoss(z, sldj, means, log_sds)
    # con_loss = criterion.conLoss(log_p_all, exp)
    con_loss = criterion.kl_div(means, log_sds, exp, log_p_all)
    con_loss_mean = con_loss.mean()

    total_con_loss += con_loss.tolist()
    # total_con_loss.append(con_loss_mean.item())
    total_nll_loss.append(nll_loss.item())

  avg_con_loss = sum(total_con_loss)/len(total_con_loss)
  avg_nll_loss = sum(total_nll_loss)/len(total_nll_loss)

  return avg_con_loss, avg_nll_loss

if __name__ == "__main__":
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")

  # SET TENSORBOARD PATH
  writer = SummaryWriter(f'./runs/{args.config}')

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)
  

  model = LatentModel(cfg)
  model = model.to(device)
  print("number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
  # PREPARE LOADER
  train_loader, val_loader = prepare_dataset(cfg)
  
  # PREPARE OPTIMIZER
  optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  # optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  scheduler = CosineAnnealingLR(optimizer, cfg.LR.T_MAX, cfg.LR.MIN_LR)

  criterion = FlowConLoss(cfg, device)

  # START TRAINING
  min_loss = 1e5
  n_bins = 0
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for i in pbar:

    # TRAINING
    avg_con_loss, avg_nll_loss, log_p, logvars = train(train_loader, i, model, optimizer, criterion, cfg, n_bins, device)
    with torch.no_grad():
      val_con_loss, val_nll_loss = validate(val_loader, model, criterion, cfg, n_bins, device)
    
    # ADJUST LR
    if cfg.LR.ADJUST:
      # scheduler.step()
      if i > cfg.LR.WARM_ITER:
        scheduler.step()
        # adjust_learning_rate(cfg, optimizer, epoch)

    curr_lr = optimizer.param_groups[0]["lr"] 

    # BEST MODEL
    if val_con_loss < min_loss:
      min_loss = val_con_loss
      best_epoch = i
      torch.save(model.state_dict(), f"checkpoints/{args.config}_model_final.pt")

    # SAVE MODEL EVERY k EPOCHS
    # if i % 200 == 0:
    #   torch.save(model.state_dict(), f"checkpoint/{args.config}_model_{str(i + 1).zfill(6)}.pt")
      # torch.save(optimizer.state_dict(), f"checkpoint/{args.config}_optim_{str(i + 1).zfill(6)}.pt")

    pbar.set_description(
      f"Train NLL Loss: {round(avg_nll_loss, 4):.5f}; Train Con Loss: {round(avg_con_loss, 4)};\
        Val NLL Loss: {round(val_nll_loss, 4)}; Val Con Loss: {round(val_con_loss, 4)}\
        logP: {log_p.item():.5f}; lr: {curr_lr:.7f}; Min NLL: {round(min_loss, 3)} ({best_epoch})"
        f"logvars: {round(logvars[0].item(), 4), round(logvars[1].item(), 4)}"
      )
    
    writer.add_scalar("Train/Contrastive", round(avg_con_loss, 4), i)
    writer.add_scalar("Train/NLL", round(avg_nll_loss, 4), i)
    writer.add_scalar("Val/Contrastive", round(val_con_loss, 4), i)
    writer.add_scalar("Val/NLL", round(val_nll_loss, 4), i)
    writer.add_scalar("lr", round(curr_lr, 7), i)
    
