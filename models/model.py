import torch 
from torch import nn 
import torchvision.models as models

import timm
from sys import exit as e
from icecream import ic

from .realnvp import RealNVPTabular

class LatentModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    # self.effnet = timm.create_model('tf_efficientnet_b0_ns', pretrained=False, num_classes=0, global_pool='')

    if cfg.TRAINING.PRETRAINED == "eff":
      self.effnet = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
      self.effnet.classifier = nn.Identity()
      self.effnet.load_state_dict(torch.load('./checkpoints/state_vggface2_enet0_new.pt', map_location=torch.device('cpu')))
      for param in self.effnet.parameters():
        param.requires_grad=False
      self.effnet.eval()
    elif cfg.TRAINING.PRETRAINED == "res":
      resnet = models.resnet18(True)
      self.effnet = nn.Sequential(*list(resnet.children())[:-1])

      pretrained = torch.load("./checkpoints/pretrained/Resnet18_MS1M_pytorch.pth.tar")
      pretrained_state_dict = pretrained['state_dict']
      model_state_dict = resnet.state_dict()
      loaded_keys = 0
      total_keys = 0
      for key in pretrained_state_dict:
          if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
              pass
          else:    
              model_state_dict[key] = pretrained_state_dict[key]
              total_keys+=1
              if key in model_state_dict:
                  loaded_keys+=1
      # resnet.load_state_dict(model_state_dict, strict = False)  
      self.effnet.load_state_dict(model_state_dict, strict = False)  
      for param in self.effnet.parameters():
        param.requires_grad=False
      self.effnet.eval()

    self.flow = RealNVPTabular(in_dim=cfg.FLOW.IN_FEAT, hidden_dim=cfg.FLOW.MLP_DIM, num_layers=cfg.FLOW.N_FLOW, \
                    num_coupling_layers=cfg.FLOW.N_BLOCK, init_zeros=cfg.FLOW.INIT_ZEROS)
    
  def forward(self, x):
    x = self.effnet(x).squeeze()
    x, mean, log_sd, logdet = self.flow(x)
    return x, mean, log_sd, logdet



