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

from models import RealNVPTabular
from utils import plot_umap, gaussian_log_p, gaussian_sample, get_args, seed_everything
from configs import get_cfg_defaults
# from losses import FlowConLoss

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

def train(loader, epoch, model, optimizer, criterion, cfg, n_bins, device):
  avg_con_loss = []
  avg_nll_loss = []
  
  model.train()
  for b, (image, exp, _) in enumerate(loader,0):
    if cfg.DATASET.AUG2:
      image = torch.cat(image, dim=0)
      exp = exp.repeat(2)
    # with torch.no_grad():
    #   plot_loader_imgs(image, exp, cfg)
    
    image = image.to(device)
    exp = exp.to(device)

    if cfg.FLOW.N_BITS < 8:
      image = torch.floor(image / 2 ** (8 - cfg.FLOW.N_BITS))
    image = image / n_bins - 0.5

    # WARMUP LEARNING RATE
    warmup_learning_rate(cfg, epoch, b, len(loader), optimizer)

    # FORWARD 
    means, log_sds, logdet, out, log_p = model(image + torch.rand_like(image) / n_bins)

    # LOSS
    nll_loss, log_p, _, log_p_all = criterion.nllLoss(out, logdet, means, log_sds)
    con_loss = criterion.conLoss(log_p_all, exp)
    con_loss_mean = con_loss.mean()
    loss = con_loss_mean + (cfg.TRAINING.LMBD * nll_loss)

    with torch.no_grad():
      avg_con_loss += con_loss.tolist()
      avg_nll_loss.append(nll_loss.item())
      # avg_cls_loss.append(class_loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
  avg_con_loss = sum(avg_con_loss)/len(avg_con_loss)
  avg_nll_loss = sum(avg_nll_loss)/len(avg_nll_loss)
  # avg_cls_loss = sum(avg_cls_loss)/len(avg_cls_loss)

  return avg_con_loss, avg_nll_loss, log_p

def validate(loader, model, criterion, cfg, n_bins, device):
  total_con_loss = []
  total_nll_loss = []
  model.eval()
  for b, (image, exp, _) in enumerate(loader,0):
      
    image = image.to(device)
    exp = exp.to(device)

    if cfg.FLOW.N_BITS < 8:
      image = torch.floor(image / 2 ** (8 - cfg.FLOW.N_BITS))
    image = image / n_bins - 0.5

    means, log_sds, logdet, out, _ = model(image + torch.rand_like(image) / n_bins)

    nll_loss, _, _, log_p_all = criterion.nllLoss(out, logdet, means, log_sds)
    con_loss = criterion.conLoss(log_p_all, exp)

    total_con_loss += con_loss.tolist()
    total_nll_loss.append(nll_loss.item())

  avg_con_loss = sum(total_con_loss)/len(total_con_loss)
  avg_nll_loss = sum(total_nll_loss)/len(total_nll_loss)

  return avg_con_loss, avg_nll_loss

def sample(net, prior, batch_size, eps, cls=None):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    mean = torch.zeros((256, 2))
    log_sd = torch.zeros((256,2))

    with torch.no_grad():
        if cls is not None:
            z = prior.sample((batch_size,), gaussian_id=cls)
        else:
            # z = prior.sample((batch_size,))
            eps = torch.randn((256, 2))
            z = gaussian_sample(eps, mean, log_sd)
        x = net.inverse(z)

        return x

class DatasetMoons:
    """ two half-moons """
    def sample(self, n):
        moons, y = datasets.make_moons(n_samples=n, noise=0.05)
        moons = moons.astype(np.float32)
        y = y.astype(np.int8)
        return torch.from_numpy(moons), torch.from_numpy(y)

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
  
  # epochs = 2001
  # batch = 256
  # feature_dim = 2
  # num_mid_channels = 32
  # num_blocks = 3
  # num_coupling_layers_per_scale = 8
  # init_zeros = False

  prior = TransformedDistribution(Uniform(torch.zeros(cfg.FLOW.IN_FEAT), torch.ones(cfg.FLOW.IN_FEAT)), SigmoidTransform().inv)

  model = RealNVPTabular(in_dim=cfg.FLOW.IN_FEAT, hidden_dim=cfg.FLOW.MLP_DIM, num_layers=cfg.FLOW.N_FLOW,
                    num_coupling_layers=cfg.FLOW.N_BLOCK, init_zeros=cfg.FLOW.INIT_ZEROS)
  model = model.to(device)
    # optimizer
  optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WT_DECAY) # todo tune WD
  print("number of params: ", sum(p.numel() for p in model.parameters()))

  # DATASET
  d = DatasetMoons()

  # criterion = FlowConLoss()
  model.train()
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for k in pbar:
      x, y = d.sample(cfg.TRAINING.BATCH)
      x = x.to(device)

      z, mean, log_sd = model(x)
      sldj = model.logdet()
      # prior_logprob = prior.log_prob(z).view(x.size(0), -1).sum(1)
      prior_logprob = gaussian_log_p(z, mean, log_sd).sum(-1)
      logprob = prior_logprob + sldj
      loss = -torch.sum(logprob) # NLL

      model.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        if k % 100 ==0:
          ic(loss.item())
          x_rec = sample(model, prior, z, cfg.TRAINING.BATCH)
          plt.scatter(x_rec[:,0], x_rec[:,1], c='g', s=5)
          plt.savefig(f"./data/x_{k}.png")
          plt.close()
          plt.scatter(z[:,0], z[:,1], c='r', s=5)
          plt.savefig(f"./data/z_{k}.png")
          plt.close()
      
        if k == cfg.TRAINING.ITER - 1:
          new_z = torch.cat([z, prior_logprob.view(-1, 1)], dim=-1)
          plot_umap(new_z, dim=3)
