import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b3
import torch.nn.functional as F

import sys

data_folder = 'data/'
# Define any image preprocessing steps you want to apply
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])