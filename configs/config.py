from yacs.config import CfgNode as CN

_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.DATA_ROOT = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.VIS_PATH = ""

# FLOW
_C.FLOW = CN()
_C.FLOW.N_FLOW = 8
_C.FLOW.N_BLOCK = 1
_C.FLOW.IN_FEAT = 2
_C.FLOW.MLP_DIM = 23
_C.FLOW.N_BITS = 5
_C.FLOW.N_BINS = 32
_C.FLOW.TEMP = 0.7
_C.FLOW.N_SAMPLE = 20
_C.FLOW.INIT_ZEROS = False

# DATASET
_C.DATASET = CN()
_C.DATASET.DS_NAME = "BU3D"
_C.DATASET.N_CLASS = 2
_C.DATASET.IMG_SIZE=224
_C.DATASET.NUM_WORKERS=1
_C.DATASET.TEST = "M0008"
_C.DATASET.COUNT = 3000
_C.DATASET.N_VIEWS = 2
_C.DATASET.AUG = False
_C.DATASET.AUG2 = False
_C.DATASET.W_SAMPLER = False

# LOSS
_C.LOSS = CN()
_C.LOSS.LAMBDA = 0.0051

# TRAINING
_C.TRAINING = CN()
_C.TRAINING.ITER = 1000
_C.TRAINING.BATCH = 256
_C.TRAINING.LR = 1e-3
_C.TRAINING.WT_DECAY = 1e-5
_C.TRAINING.LMBD = 2.
_C.TRAINING.OUTPUT_DIM = 3072
_C.TRAINING.DROPOUT = False
_C.TRAINING.TEMP = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
_C.TRAINING.PRETRAINED = "eff"

# LEARNING RATE
_C.LR = CN()
_C.LR.WARM = False
_C.LR.ADJUST = False
_C.LR.WARM_ITER = 10
_C.LR.WARMUP_FROM = 1e-5
_C.LR.DECAY_RATE = 0.1
_C.LR.MIN_LR = 1e-6
_C.LR.T_MAX = 100

# COMMENTS
_C.COMMENTS = "TEST"




def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()