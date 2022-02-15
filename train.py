__author__ = "Yuhao Liu"

import os
import os.path as p
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from ternausnet.models import UNet16
from config import CUDA_DEVICE

"""Global Parameters"""
version = None  # defined in main
model_name = None  # defined in main


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))


def main():
    global version, model_name
    version = "-v0.0.0"
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + version)
    net = UNet16()
    # train
    tb.close()


if __name__ == "__main__":
    main()
