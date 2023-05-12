import torch 
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torchvision import utils
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

import sys
sys.path.append('.')
from einops import rearrange

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

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

def get_features(loader, model, cfg, device):
  robust = False
  model.eval()
  features = []
  labels = []
  for b, (image, exp, _) in enumerate(loader,0):
    image = image.to(device)
    exp = exp.to(device)
    z, *_ = model(image)
    features.append(z)
    labels.append(exp)

  features = torch.cat(features, dim=0)
  labels = torch.cat(labels, dim=0)
  return features.cpu().numpy(), labels.cpu().numpy()

if __name__ == "__main__":
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  config_path = os.path.join("./configs/experiments", f"{args.config}.yaml")

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.DATASET.W_SAMPLER = False
  cfg.DATASET.AUG = False
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
  print("number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

  # PREPARE LOADER
  train_loader, val_loader, _ = prepare_dataset(cfg)

  with torch.no_grad():
    X_train, y_train = get_features(train_loader, model, cfg, device)
    X_val, y_val = get_features(val_loader, model, cfg, device)
  
  # # # clf = GaussianNB()
  clf = RandomForestClassifier(n_estimators=200, criterion='gini', class_weight='balanced', random_state=42)
  clf.fit(X_train, y_train)
  y_pred_val = clf.predict(X_val)
  ic(get_metrics(y_val, y_pred_val))
  

  dist_path = f"./data/distributions/{args.config}"
  mu = torch.load(os.path.join(dist_path, "mu.pt"))
  mu =  torch.stack(mu)
  log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"))
  log_sd =  torch.stack(log_sd)

  dist = Normal(mu, log_sd.exp())
  sample = dist.sample(torch.Size([1000]))

  sample = sample[:, 1:3, :]
  sample = rearrange(sample, 'b c d -> (b c) d')
  # z_train_fear = sample[:, 1]
  X_train_rec = model.flow.inverse(sample).detach().cpu()

  # ic(X_train_rec.size())
  # X_train_rec = rearrange(X_train_rec, '(b c) d -> b c d', c=7)
  # ic(X_train_rec.size())
  # e()

  X_train_rec = X_train_rec.numpy()
  y_train_rec = torch.tensor([1,2]).repeat(1000).numpy()
  # y_train_fear = torch.tensor([1]).repeat(300).numpy()

  X_train_new = np.concatenate((X_train, X_train_rec))
  y_train_new = np.concatenate((y_train, y_train_rec))

  p = np.random.permutation(len(X_train_new))
  X_train_new = X_train_new[p]
  y_train_new = y_train_new[p]

  clf = RandomForestClassifier(n_estimators=200, criterion='gini', class_weight='balanced', random_state=42)
  clf.fit(X_train_new, y_train_new)
  y_pred_val = clf.predict(X_val)
  ic(get_metrics(y_val, y_pred_val))
  e()





    
