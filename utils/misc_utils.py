import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from umap import UMAP
import pandas as pd
import plotly.express as px

import random
import torch
import argparse
from math import log, pi, exp

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

from sys import exit as e
from icecream import ic



inv_exp_dict = {
                  "AFF": {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger"},
                  "RAF": {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}, 
                  "BU3D": {0: "Neutral", 1: "Angry", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprise"}
                }

def mkdir(path):
  if not os.path.isdir(path):
    os.mkdir(path)

# def plot_umap(cfg, X_lst_un, y_lst, name, fname_all, dim, mode):
#   b = X_lst_un.size(0)
#   X_lst = UMAP(n_components=dim, random_state=0, init='random').fit_transform(X_lst_un.view(b, -1))
#   y_lst_label = [str(i) for i in y_lst.detach().numpy()]

#   if dim == 3:
#     df = pd.DataFrame(X_lst, columns=["x", "y", "z"])
#   else:
#     df = pd.DataFrame(X_lst, columns=["x", "y"])
#   df_color = pd.DataFrame(y_lst_label, columns=["class"])
#   df_fname = pd.DataFrame(fname_all, columns=["fname"])
#   df = df.join(df_color)
#   df = df.join(df_fname)
#   cls_list  = list(inv_exp_dict[cfg.DATASET.DS_NAME].values())
#   cls_lst_name = [cls_list[int(k)] for k in y_lst_label]
#   df_exp = pd.DataFrame(cls_lst_name, columns=["expr"])
#   df = df.join(df_exp)
  
#   if dim == 3:
#     fig = px.scatter_3d(df, x='x', y='y', z='z',color='class', title=f"{name}", \
#       # hover_data=[df.fname])
#       category_orders={"class": list(inv_exp_dict[cfg.DATASET.DS_NAME].values())}, hover_data=[df.fname, df.expr])
#   else:
#     fig = px.scatter(df, x='x', y='y',color='class', title=f"{name}", \
#       # hover_data=[df.fname])
#       category_orders={"class": list(inv_exp_dict[cfg.DATASET.DS_NAME].values())}, hover_data=[df.fname, df.expr])
  
#   fig.update_traces(marker=dict(size=6))
#   fig.update_layout(legend=dict(
#     yanchor="top",
#     y=0.60,
#     xanchor="left",
#     x=0.70
#     ))
  
#   dest_path = os.path.join("./data/umap/", name)
#   mkdir(dest_path)
#   # fig.update_traces(hovertemplate = 'fname=%{customdata[0]}<br>')
#   fig.write_html(os.path.join(dest_path, f"{dim}d_{mode}.html"))

def grad_flow(named_parameters):
  ave_grads = 0
  layers_cnt = 0
  for n, p in named_parameters:
    if(p.requires_grad) and ("bias" not in n):
      ave_grads += p.grad.abs().mean()
      layers_cnt += 1
  ave_grads /= layers_cnt
  return ave_grads
  
def plot_loader_imgs(arr, exp, cfg):
  arr = arr.permute(0, 2, 3, 1)
  arr = arr.detach().cpu().numpy()
  b, h, w, c = arr.shape
  for i in range(b):
    img = arr[i]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (img*255).astype(np.uint8)

    expression = exp[i].item()
    img = cv2.putText(img, str(inv_exp_dict[cfg.DATASET.DS_NAME][expression]), (int(h-100), int(w-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 2)

    cv2.imwrite(os.path.join(cfg.PATHS.VIS_PATH, f"img_{i}.png"), img)

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--config', type=str, default='default', help='configuration to load')
  args = parser.parse_args()
  return args


def get_metrics(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  mcc = matthews_corrcoef(y_true, y_pred)
  conf = confusion_matrix(y_true, y_pred)
  return round(acc, 3), round(mcc, 2), conf

def gaussian_log_p(x, mean, log_sd):
  return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
  return mean + torch.exp(log_sd) * eps
  # return mean + torch.exp(log_sd) 


def plot_umap(X_lst, dim=2):
  b = X_lst.size(0)
  # X_lst = UMAP(n_components=dim, random_state=0, init='random').fit_transform(X_lst_un.view(b, -1))

  if dim == 3:
    df = pd.DataFrame(X_lst, columns=["x", "y", "z"])
  else:
    df = pd.DataFrame(X_lst, columns=["x", "y"])
  
  if dim == 3:
    fig = px.scatter_3d(df, x='x', y='y', z='z',title=f"test", \
      # hover_data=[df.fname])
      # category_orders={"class": list(inv_exp_dict[cfg.DATASET.DS_NAME].values())}, hover_data=[df.fname, df.expr]
        )
  else:
    fig = px.scatter(df, x='x', y='y', title=f"test", \
      # hover_data=[df.fname])
      # category_orders={"class": list(inv_exp_dict[cfg.DATASET.DS_NAME].values())}, hover_data=[df.fname, df.expr]
      )
  
  fig.update_traces(marker=dict(size=6))
  fig.update_layout(legend=dict(
    yanchor="top",
    y=0.60,
    xanchor="left",
    x=0.70
    ))
  
  fig.write_html("plot.html")