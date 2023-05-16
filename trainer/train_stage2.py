import torch 
from torch import nn, optim
from torch.nn import functional as F
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


class LinearClassifier(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.linear = nn.Linear(cfg.FLOW.IN_FEAT, 256)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(0.3)
    self.linear2 = nn.Linear(256, cfg.DATASET.N_CLASS)
    
  def forward(self, x):
    x = self.linear(x)
    x = self.drop(x)
    x = self.relu(x)
    x = self.linear2(x)
    return F.softmax(x, dim=-1)


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


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target, weights):
    #logsoftmax = nn.LogSoftmax(dim=-1)
    return torch.mean(torch.sum(- weights*soft_target * torch.nn.functional.log_softmax(pred, -1), 1))

def cross_entropy_with_label_smoothing(pred, target, weights):
    soft_target = label_smooth(target, pred.size(1)) #num_classes) #
    return cross_entropy_loss_with_soft_target(pred, soft_target, weights)



def prepare_dataset(cfg):
  loss_wts = None
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
      loss_wts = weight/np.amin(weight)
    else:
      sampler=None
      train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
        num_workers=cfg.DATASET.NUM_WORKERS)
    
    val_dataset = RafDb(cfg, "val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
  
  class_freq = np.array(list(train_dataset.cnt_dict.values()))
  weight = 1./class_freq
  loss_wts = weight/np.amin(weight)

  return train_loader, val_loader, loss_wts

def train(loader, epoch, model, classifier, optimizer, criterion, cfg, device, sample_weight):
  robust = False
  avg_loss = []
  y_pred = []
  y_true = []
  model.eval()
  classifier.train()
  for b, (image, exp, _) in enumerate(loader,0):
    if cfg.DATASET.AUG2:
      image = torch.cat(image, dim=0)
      exp = exp.repeat(2)
    # with torch.no_grad():
    #   plot_loader_imgs(image, exp, cfg)
    
    image = image.to(device)
    exp = exp.to(device)

    # WARMUP LEARNING RATE
    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)

    # FORWARD 
    with torch.no_grad():
      z, *_ = model(image)
    out = classifier(z)

    loss = criterion(out, exp, sample_weight)

    if robust:
      #optimizer.zero_grad()
      loss.backward()
      optimizer.first_step(zero_grad=True)
      # second forward-backward pass
      with torch.no_grad():
        z, *_ = model(image)
      out = classifier(z)
      loss = criterion(out, exp, sample_weight)
      loss.backward()
      optimizer.second_step(zero_grad=True)
    else:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    with torch.no_grad():
      avg_loss.append(loss.item())
      y_pred += torch.argmax(out, dim=-1).cpu().tolist()
      y_true += exp.cpu().tolist()      

  avg_loss = sum(avg_loss)/len(avg_loss)
  return avg_loss, y_pred, y_true

def validate(loader, model, classifier, criterion, device, sample_weight):
  total_loss = []
  y_pred = []
  y_true = []
  model.eval()
  classifier.eval()
  for b, (image, exp, _) in enumerate(loader,0):
      
    image = image.to(device)
    exp = exp.to(device)

    # FORWARD 
    z, *_ = model(image)
    out = classifier(z)
    loss = criterion(out, exp, sample_weight)

    total_loss.append(loss.item())
    y_pred += torch.argmax(out, dim=-1).cpu().tolist()
    y_true += exp.cpu().tolist()

  total_loss = sum(total_loss)/len(total_loss)
  return total_loss, y_pred, y_true

if __name__ == "__main__":
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")

  # SET TENSORBOARD PATH
  writer = SummaryWriter(f'./runs2/{args.config}_stage2')

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.DATASET.W_SAMPLER = False
  cfg.TRAINING.LR = 1e-3
  cfg.TRAINING.ITER = 101
  # cfg.LR.ADJUST = True
  # cfg.LR.WARM = False
  cfg.freeze()
  print(cfg)
  

  model = LatentModel(cfg)
  model = model.to(device)
  checkpoint = torch.load(f"./checkpoints/{args.config}_model_final.pt", map_location=device)
  model.load_state_dict(checkpoint)

  classifier = LinearClassifier(cfg)
  classifier = classifier.to(device)
  print("number of params: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

    # optimizer

  # PREPARE LOADER
  train_loader, val_loader, loss_weights = prepare_dataset(cfg)
  loss_weights = torch.from_numpy(loss_weights).to(device)
  
  # PREPARE OPTIMIZER
  optimizer = optim.AdamW(classifier.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  # optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, classifier.parameters()), optim.Adam, lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY)
  scheduler = CosineAnnealingLR(optimizer, cfg.LR.T_MAX, cfg.LR.MIN_LR)

  # criterion = nn.CrossEntropyLoss()
  criterion = cross_entropy_with_label_smoothing

  # START TRAINING
  min_loss = 1e5
  best_acc = 0
  n_bins = 0
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for i in pbar:

    # TRAINING
    train_loss, y_pred_train, y_true_train = train(train_loader, i, model, classifier, optimizer, criterion, cfg, device, loss_weights)
    with torch.no_grad():
      val_loss, y_pred_val, y_true_val = validate(val_loader, model, classifier, criterion, device, loss_weights)
    
    train_acc, train_mcc, train_conf = get_metrics(y_pred_train, y_true_train)
    val_acc, val_mcc, val_conf = get_metrics(y_pred_val, y_true_val)
    # ADJUST LR
    if cfg.LR.ADJUST:
      scheduler.step()

    curr_lr = optimizer.param_groups[0]["lr"] 


    # BEST MODEL
    if val_acc > best_acc:
      min_loss = val_loss
      best_acc = val_acc
      ic(val_conf)
      torch.save(model.state_dict(), f"checkpoints/{args.config}_model_final_linear.pt")

    # SAVE MODEL EVERY k EPOCHS
    # if i % 200 == 0:
    #   torch.save(model.state_dict(), f"checkpoint/{args.config}_model_{str(i + 1).zfill(6)}.pt")
      # torch.save(optimizer.state_dict(), f"checkpoint/{args.config}_optim_{str(i + 1).zfill(6)}.pt")

    
    pbar.set_description(
        f"train_loss: {round(train_loss, 4)}; train_acc: {round(train_acc, 4)};"
        f"val_loss: {round(val_loss, 4)}; val_acc: {round(val_acc, 4)};"
        f"best_acc: {round(best_acc, 4)};"
                        ) 
    
    writer.add_scalar("Train/Loss", round(train_loss, 4), i)
    writer.add_scalar("Train/Acc", round(train_acc, 4), i)
    writer.add_scalar("Val/Loss", round(val_loss, 4), i)
    writer.add_scalar("Val/Acc", round(val_acc, 4), i)
    
