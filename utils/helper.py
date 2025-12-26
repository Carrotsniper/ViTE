
import random

import yaml
import torch
import numpy as np

from box import Box


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        opts = yaml.safe_load(f)
    opts = Box(opts)
    return opts
