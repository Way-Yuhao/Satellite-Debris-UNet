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
from customDataFolderInria import ImageFolderInria

"""Global Parameters"""
indira_dataset_path = "/mnt/data1/yl241/datasets/Inria_Aerial/AerialImageDataset/"
network_weight_path = "./weight/"
version = None  # defined in main
model_name = None  # defined in main
num_workers_train = 8
batch_size = 8

"Hyper Parameters"
init_lr = 5e-3
epoch = 2000


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))


def load_data(dataset_path):
    data_loader = torch.utils.data.DataLoader(
        ImageFolderInria(root=dataset_path),
        batch_size=batch_size, num_workers=num_workers_train)
    return data_loader


def load_network_weights(net, path):
    raise NotImplementedError


def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, label):
    criterion = nn.BCELoss()
    bce_loss = criterion(output, label)
    return bce_loss


def disp_plt(img, title="", idx=None):
    """
    :param img: image to display
    :param title: title of the figure
    :param idx: index of the file, for print purposes
    :param tone_map: applies tone mapping via cv2 if set to True
    :return: None
    """
    img = img.detach().clone()

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    if img.shape[1] == 3:  # RGB
        img = img.cpu().squeeze().permute(1, 2, 0)
    else:  # monochrome
        img = img.cpu().squeeze()
        img = torch.stack((img, img, img), dim=0).permute(1, 2, 0)
    img = np.float32(img)
    plt.imshow(img)
    # compiling title
    if idx:
        title = "{} (index {})".format(title, idx)
    full_title = "{} / {}".format(version, title)
    plt.title(full_title)
    return


def tensorboard_vis(tb, ep):
    raise NotImplementedError()


def train(net, tb, load_weights, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()
    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    train_loader = load_data(p.join(indira_dataset_path, "train"))
    train_num_mini_batches = len(train_loader)
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=.5)

    running_train_loss = 0.0
    train_output = None
    for ep in range(epoch):
        print("Epoch ", ep)
        train_iter = iter(train_loader)
        for _ in tqdm(range(train_num_mini_batches)):
            input_, label = train_iter.next()
            input_, label = input_.to(CUDA_DEVICE), label.to(CUDA_DEVICE)
            # optimizer.zero_grad()
            train_output = net(input_)
            train_loss = compute_loss(train_output, label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        print("train loss = {:.4}".format(cur_train_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)

    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def main():
    global version, model_name
    model_name, version = "unet16", "-v0.0.0"
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + version)
    net = UNet16()
    train(net, tb, load_weights=False, pre_trained_params_path=None)
    tb.close()


if __name__ == "__main__":
    main()
