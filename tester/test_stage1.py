import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import utils
from PIL import Image

from einops import rearrange, reduce, repeat

from torch.nn import functional as F
import os
import random
import numpy as np
from sys import exit as e
from tqdm import tqdm
from math import log, sqrt, pi
from icecream import ic
import argparse

import albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2
import sys
sys.path.append('.')

from models import LatentModel
from losses import FlowConLoss
from configs import get_cfg_defaults
from utils import seed_everything, get_args




def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def nllLoss(cfg, device, z, logdet, mu, log_sd):
  # allowed_labels = [0, 3, 4]
  n_bins = 32
  n_pixel = cfg.FLOW.IN_FEAT
  init_loss = -log(n_bins) * n_pixel
  # b_size, _, _, _ = out[0].size()
  b_size = cfg.DATASET.N_CLASS #len(allowed_labels)

  # Calculate total log_p
  log_p_total = 0
  log_p_all = torch.zeros((b_size), dtype=torch.float32, device=device)
  log_p_nll = 0

  # Create mask to select NLL loss elements
  b, d= z.size()
  # z = z.view(b, 1, c, h, w)
  nll_mask = torch.eye(b, device=device).view(b, b, 1)
  nll_mask = nll_mask.repeat(1, 1, d)


  # Square matrix for contrastive loss evaluation      
  log_p_batch = gaussian_log_p(z, mu, log_sd)
  # NLL losses
  log_p_nll_block = (log_p_batch * nll_mask).sum(dim=(2))
  log_p_nll_block = log_p_nll_block.sum(dim=1)
  log_p_nll += log_p_nll_block

  log_p_all += log_p_batch.sum(dim=(1))

  logdet = logdet.mean()
  loss = init_loss + logdet + log_p_nll
  return ( 
    (-loss / (log(2) * n_pixel)).mean(),
    (log_p_nll / (log(2) * n_pixel)).mean(),
    (logdet / (log(2) * n_pixel)).mean(), 
    (log_p_all/ (log(2) * n_pixel))
    # log_p_nll
  )


def calc_likelihood(cfg, z, mu, log_sd, device, n_pixel):
    z = rearrange(z, 'b d -> b 1 d')
    log_p_batch = gaussian_log_p(z, mu, log_sd)
    log_p_all = log_p_batch.sum(dim=(2))
    return log_p_all/ (log(2) * n_pixel)

def get_image_tensor(path):
  transforms = A.Compose([
      A.Resize(cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE, p=1),
      A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
      ToTensorV2()
      ], 
      # keypoints_params = A.KeypointParams(format='xy'), 
      p=1)
  train_image = np.array(Image.open(path))
  # train_image = np.array(Image.open("./data/test/F0020_SA03WH_F2D.bmp"))
  train_image = transforms(image = train_image)["image"].unsqueeze(0)
  return train_image

def get_ll(image, means, log_sds, model):
  # NLL & CONTRASTIVE LOSS
  image = image.type(torch.float32)
  features, *_  = model(image)
  log_probs = F.softmax(calc_likelihood(cfg, features, mu, log_sd, device, n_pixel), dim=-1)
  # z, _, _, sldj  = model(image)
  # nll_loss, log_p, log_det, log_p_all = nllLoss(cfg, device, z, sldj, means, log_sds)
  return log_probs




if __name__ == "__main__":
  seed_everything(42)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)

  n_pixel = cfg.FLOW.IN_FEAT
  model = LatentModel(cfg)
  model = model.to(device)

  checkpoint = torch.load(f"./checkpoints/{args.config}_model_final.pt", map_location=device)
  # checkpoint = torch.load(f"./checkpoint/{args.config}_model_000201.pt", map_location=device)
  model.load_state_dict(checkpoint)

  # train_image = get_image_tensor("/data/dataset/BU3DFE/M0007/M0007_HA01AE_F2D.bmp")
  # train_image = get_image_tensor("/data/dataset/BU3DFE/F0030/F0030_FE01BL_F2D.bmp")
  
  train_image = get_image_tensor("/data/dataset/raf_db/basic/Image/aligned/train_02089_aligned.jpg")
  train_image = train_image.to(device)
  # train_image = get_image_tensor("/data/dataset/raf_db/basic/Image/aligned/train_08604_aligned.jpg")
  # train_image = get_image_tensor("/data/dataset/raf_db/basic/Image/aligned/test_0286_aligned.jpg")
  dist_path = f"./data/distributions/{args.config}"
  means = torch.load(os.path.join(dist_path, "mu.pt"))
  log_sds = torch.load(os.path.join(dist_path, "log_sd.pt"))
  mu =  torch.stack(means, dim=0)
  log_sd =  torch.stack(log_sds, dim=0)

  ic(train_image.size(), mu.size(), log_sd.size())

  train_ll = get_ll(train_image, mu, log_sd, model)
  print("-"*40)
  ic(train_ll)





  

  
  
