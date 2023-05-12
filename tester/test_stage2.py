import torch 
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import utils
from PIL import Image

import os
import random
import numpy as np
from sys import exit as e
from tqdm import tqdm
from math import log, sqrt, pi
from icecream import ic
import json
import argparse

import sys
sys.path.append('.')
from einops import rearrange, reduce, repeat

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2

from models import LatentModel
from losses import FlowConLoss
from configs import get_cfg_defaults
from utils import seed_everything, get_args, get_metrics
from loaders import CustomDataset, AffectDataset, RafDb

def prepare_dataset(cfg, load_train = False):
  train_loader = None
  # PREPARE LOADER
  if cfg.DATASET.DS_NAME == "BU3D":
    if load_train:
      train_dataset = CustomDataset(cfg, "train")
      train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
        num_workers=cfg.DATASET.NUM_WORKERS)
    val_dataset = CustomDataset(cfg, "val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
  
  elif cfg.DATASET.DS_NAME == "AFF":
    if load_train:
      train_dataset = AffectDataset(cfg, "train")
      train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
        num_workers=cfg.DATASET.NUM_WORKERS)
    val_dataset = AffectDataset(cfg, "val")
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH, shuffle=True, \
      num_workers=cfg.DATASET.NUM_WORKERS)
    
  elif cfg.DATASET.DS_NAME == "RAF":
    if load_train:
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

def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def calc_likelihood(cfg, z, mu, log_sd, device, n_pixel):
  mu =  torch.stack(mu)
  log_sd =  torch.stack(log_sd)
  # log_p_all = torch.zeros((out[-1].size(0), cfg.DATASET.N_CLASS), dtype=torch.float32, device=device)

  z = rearrange(z, 'b d -> b 1 d')
  log_p_batch = gaussian_log_p(z, mu, log_sd)
  log_p_all = log_p_batch.sum(dim=(2))
  
  return log_p_all/ (log(2) * n_pixel)

def validate(cfg, loader, n_bins, device):
  y_val_pred = []
  y_val_pred_vec = []
  y_val_true = []
  total_loss = []
  fname_all = []
  log_p = []

  n_pixel = cfg.FLOW.IN_FEAT
  dist_path = f"./data/distributions/{args.config}"
  mu = torch.load(os.path.join(dist_path, "mu.pt"))
  log_sd = torch.load(os.path.join(dist_path, "log_sd.pt"))

  for b, (image, exp, fname) in enumerate(loader,0):
    image = image.to(device)
    exp = exp.type(torch.int64).to(device)
    fname_all += list(fname)

    features, *_  = model(image)
    log_probs = calc_likelihood(cfg, features, mu, log_sd, device, n_pixel)
    # pred = torch.argmax(log_probs, dim=1).cpu().tolist()
    _, pred = torch.topk(log_probs, k=2, dim=-1)
    pred = pred.T
    target_reshaped = exp.view(1, -1).expand_as(pred)
    # new_pred = torch.where(pred == target_reshaped, target_reshaped, pred)[1]
    new_pred = torch.where((pred == target_reshaped).any(dim=0), target_reshaped, pred)[0]
    
    y_val_pred += new_pred.cpu().tolist()
    y_val_true += list(exp.cpu())

  
  val_acc, _, val_conf = get_metrics(y_val_true, y_val_pred)
  class_wise_acc = val_conf.diagonal()/val_conf.sum(axis=1)
  return val_acc, val_conf, class_wise_acc


def save_json(args, fname_all, y_val_true, y_val_pred, log_p):
  inv_exp_dict3 = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}
  metadata_dict = {"correct": {}, "incorrect": {}}

  log_p_soft = F.softmax(log_p, dim=1)

  for k in range(len(fname_all)):
    fname = fname_all[k]
    if torch.argmax(y_val_pred[k]).item() == y_val_true[k].item():
      metadata_dict["correct"][fname] = {}
      metadata_dict["correct"][fname]["exp"] = inv_exp_dict3[y_val_true[k].item()]
      metadata_dict["correct"][fname]["class"] = str(y_val_true[k].item())

      metadata_dict["correct"][fname]["y_prob"] = json.dumps(str(list(np.round(y_val_pred[k].cpu().numpy(), 4))))
      metadata_dict["correct"][fname]["likelihood"] = json.dumps(str(list(np.round(log_p_soft[k].cpu().numpy(), 4))))
      # metadata_dict[fname]["accurate"] = "Yes" if torch.argmax(y_val_pred[k]).item() == y_val_true[k].item() else "No"

      metadata_dict["correct"][fname]["predicted"] = inv_exp_dict3[torch.argmax(y_val_pred[k]).item()]
    else:
      metadata_dict["incorrect"][fname] = {}
      metadata_dict["incorrect"][fname]["exp"] = inv_exp_dict3[y_val_true[k].item()]
      metadata_dict["incorrect"][fname]["class"] = str(y_val_true[k].item())

      metadata_dict["incorrect"][fname]["y_prob"] = json.dumps(str(list(np.round(y_val_pred[k].cpu().numpy(), 4))))
      metadata_dict["incorrect"][fname]["likelihood"] = json.dumps(str(list(np.round(log_p_soft[k].cpu().numpy(), 4))))
      # metadata_dict[fname]["accurate"] = "Yes" if torch.argmax(y_val_pred[k]).item() == y_val_true[k].item() else "No"

      metadata_dict["incorrect"][fname]["predicted"] = inv_exp_dict3[torch.argmax(y_val_pred[k]).item()]


  with open(f'./data/ll.json', 'w') as fp:
  # with open(f'./data/{args.config}.json', 'w') as fp:
    json.dump(metadata_dict, fp, indent=4)
  

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
  cfg.DATASET.AUG = False
  cfg.DATASET.W_SAMPLER = False
  cfg.freeze()
  print(cfg)

  n_bins = 32


  model = LatentModel(cfg)
  model = model.to(device)
  checkpoint = torch.load(f"./checkpoints/{args.config}_model_final.pt", map_location=device)
  model.load_state_dict(checkpoint)
  print("Total Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

  # PREPARE LOADER
  train_loader, val_loader = prepare_dataset(cfg, True)

  with torch.no_grad():
    res = validate(cfg, train_loader, n_bins, device)
    res2 = validate(cfg, val_loader, n_bins, device)
    ic(res)
    ic(res2)
      


  e()

  

