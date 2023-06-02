import torch 
from torch import nn 
from torch.nn import functional as F
import torchvision.models as models
from torch.utils import model_zoo


import timm
from sys import exit as e
from icecream import ic

from .realnvp import RealNVPTabular
from .inception_v1 import InceptionResnetV1
from .resnet import resnet18
from .vgg import VGG

class LatentModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg
    # BACKBONE
    if cfg.TRAINING.PRETRAINED == "eff":
      self.backbone = timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
      self.backbone.classifier = nn.Identity()
      self.backbone.load_state_dict(torch.load('./checkpoints/state_vggface2_enet0_new.pt', map_location=torch.device('cpu')))
      # self.backbone = torch.load("./checkpoints/enet_b0_7.pt")
      # self.backbone = torch.load("./checkpoints/enet_b0_8_best_vgaf.pt")
      # self.backbone = torch.load("./checkpoints/enet_b0_8_best_afew.pt")
      # self.backbone.classifier = nn.Identity()
    
    elif cfg.TRAINING.PRETRAINED == "vgg":
      self.backbone = VGG('VGG19')
      checkpoint = torch.load('./checkpoints/PrivateTest_model.t7')
      self.backbone.load_state_dict(checkpoint['net'])
      self.backbone.classifier=nn.Identity()
    # 
    elif cfg.TRAINING.PRETRAINED == "eff2":
      self.backbone = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
      self.backbone.classifier = nn.Identity()
      self.backbone.load_state_dict(torch.load('./checkpoints/state_vggface2_enet2.pt', map_location=torch.device('cpu')))

    elif cfg.TRAINING.PRETRAINED == "res":
      resnet = models.resnet18(True)
      self.backbone = nn.Sequential(*list(resnet.children())[:-1])
      pretrained = torch.load("./checkpoints/pretrained/Resnet18_MS1M_pytorch.pth.tar")
      pretrained_state_dict = pretrained['state_dict']
      model_state_dict = self.backbone.state_dict()
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
      self.backbone.load_state_dict(model_state_dict, strict = False)  
    
    elif cfg.TRAINING.PRETRAINED == "inc":
      self.backbone = InceptionResnetV1(pretrained='vggface2', in_chan=3)
    
    elif cfg.TRAINING.PRETRAINED == "res34":
      self.backbone = timm.create_model('resnet34', pretrained=False)
      self.backbone.fc = nn.Linear(cfg.FLOW.IN_FEAT, cfg.DATASET.N_CLASS)
      url="https://github.com/Emilien-mipt/fer-pytorch/releases/download/0.0.1/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt"
      cp = model_zoo.load_url(url, progress=True, map_location="cpu")
      state_dict = cp["state_dict"]
      state_dict = {k.replace("model.model.", ""): v for k, v in state_dict.items()}
      self.backbone.load_state_dict(state_dict)
      self.backbone.fc = nn.Identity()
    
    elif cfg.TRAINING.PRETRAINED == "res18":  # Same as 'res' option
      # self.backbone = resnet18("")
      resnet = models.resnet18(True)
      self.backbone = nn.Sequential(*list(resnet.children())[:-1])
      msceleb_model = torch.load('./checkpoints/pretrained/Resnet18_MS1M_pytorch.pth.tar')
      state_dict = msceleb_model['state_dict']
      self.backbone.load_state_dict(state_dict, strict=False)

    # FREEZE BACKBONE
    for param in self.backbone.parameters():
        param.requires_grad=False
    self.backbone.eval()

    # FLOW MODEL
    self.flow = RealNVPTabular(cfg, in_dim=cfg.FLOW.IN_FEAT, hidden_dim=cfg.FLOW.MLP_DIM, num_layers=cfg.FLOW.N_FLOW, \
                    num_coupling_layers=cfg.FLOW.N_BLOCK, init_zeros=cfg.FLOW.INIT_ZEROS, dropout=cfg.TRAINING.DROPOUT)
    
    self.sigma1 = nn.Parameter(torch.zeros(1))
    self.sigma2 = nn.Parameter(torch.zeros(1))


  def forward(self, x):
    # x = x + torch.rand_like(x)
    x = self.backbone(x).squeeze()
    x, mean, log_sd, logdet = self.flow(x)
    return x, mean, log_sd, logdet, [self.sigma1,self.sigma2]
    # return x, mean, log_sd, logdet



