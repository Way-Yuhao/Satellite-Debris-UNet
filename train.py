__author__ = "Yuhao Liu"

import os
import os.path as p
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler
from torchmetrics import JaccardIndex
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from ternausnet.models import UNet16, UNet11
# from config import CUDA_DEVICE
from customDataFolderInria import ImageFolderInria

"""Global Parameters"""
indira_dataset_path = "/mnt/data1/yl241/datasets/Inria_Aerial/AerialImageDataset/"
network_weight_path = "./weight/"
CUDA_DEVICE = 'cpu'  # parsed in main
version = None  # defined in main
model_name = None  # defined in main
num_workers_train = 18
batch_size = 8

"Hyper Parameters"
init_lr = 5e-4
epoch = 500
PROB_THRESHOLD = .5  # for visualizing inference output


def print_params():
    print("######## Basics ##################")
    print("version: {}".format(version))
    print("Training on {}".format(CUDA_DEVICE))
    print("batch size = ", batch_size)
    print("number of workers = ", num_workers_train)
    print("#################################")


def load_data(dataset_path, sampler=None):
    data_loader = torch.utils.data.DataLoader(
        ImageFolderInria(root=dataset_path),
        batch_size=batch_size, num_workers=num_workers_train, sampler=sampler)
    return data_loader


def load_network_weights(net, path):
    raise NotImplementedError


def save_network_weights(net, ep=None):
    filename = network_weight_path + "{}{}_epoch_{}.pth".format(model_name, version, ep)
    torch.save(net.state_dict(), filename)
    print("network weights saved to ", filename)
    return


def compute_loss(output, label):
    # debug
    # assert(not torch.isnan(output).any())

    bce_criterion = nn.BCELoss()
    bce_loss = bce_criterion(output, label)
    jaccard_criterion = JaccardIndex(num_classes=2).to(CUDA_DEVICE)
    iou_loss = jaccard_criterion(output, label.type(torch.int8))
    iou_loss.requires_grad = True
    total_loss = bce_loss + iou_loss
    return total_loss


def tensorboard_vis(tb, ep, mode='train', input_=None, output=None, label=None):
    tb.add_histogram("{}/output_".format(mode), output, global_step=ep)
    tb.add_histogram("{}/label_".format(mode), label, global_step=ep)
    if input_ is not None:
        input_img_grid = torchvision.utils.make_grid(input_)
        tb.add_image("{}/input".format(mode), input_img_grid, global_step=ep)
    if output is not None:
        if mode == 'train':  # no threshold in visualization
            output_img_grid = torchvision.utils.make_grid(output)
            tb.add_image("{}/output".format(mode), output_img_grid, global_step=ep)
        elif mode == 'dev':  # apply threshold in visualization
            clipped_output = output > PROB_THRESHOLD
            output_img_grid = torchvision.utils.make_grid(clipped_output)
            tb.add_image("{}/output".format(mode), output_img_grid, global_step=ep)
    if label is not None:
        label_img_grid = torchvision.utils.make_grid(label)
        tb.add_image("{}/label".format(mode), label_img_grid, global_step=ep)
    return


def train(net, tb, load_weights, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()
    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    train_loader = load_data(p.join(indira_dataset_path, "train"))
    train_num_mini_batches = len(train_loader)
    # optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=.5)
    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    running_train_loss = 0.0
    train_input, train_output, train_label = None, None, None
    for ep in range(epoch):
        print("{}-{} | Epoch {}".format(model_name, version, ep))
        train_iter = iter(train_loader)
        for _ in tqdm(range(train_num_mini_batches)):
            train_input, train_label = train_iter.next()
            train_input, train_label = train_input.to(CUDA_DEVICE), train_label.to(CUDA_DEVICE)
            # optimizer.zero_grad()
            train_output = net(train_input)
            train_loss = compute_loss(train_output, train_label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        print("train loss = {:.4}".format(cur_train_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)

        if ep % 5 == 0:
            tensorboard_vis(tb, ep, mode='train', input_=train_input, output=train_output, label=train_label)
        running_train_loss = 0.0
    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def train_dev(net, tb, load_weights, pre_trained_params_path=None):
    print_params()
    net.to(CUDA_DEVICE)
    net.train()
    if load_weights:
        load_network_weights(net, pre_trained_params_path)

    # splitting train/dev set
    validation_split = .2
    dataset = ImageFolderInria(root=p.join(indira_dataset_path, "train"))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, dev_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)
    train_loader = load_data(p.join(indira_dataset_path, "train"), sampler=train_sampler)
    dev_loader = load_data(p.join(indira_dataset_path, "train"), sampler=dev_sampler)
    print("Using cross-validation with a {:.0%}/{:.0%} train/dev split:".format(1 - validation_split, validation_split))
    print("dev set: entry {} to {} | train set: entry {} to {}"
          .format(dev_indices[0], dev_indices[-1], train_indices[0], train_indices[-1]))
    print("size of train set = {} mini-batches | size of dev set = {} mini-batches".format(len(train_loader),
                                                                                           len(dev_loader)))
    train_num_mini_batches, dev_num_mini_batches = len(train_loader), len(dev_loader)
    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    running_train_loss, running_dev_loss = 0.0, 0.0
    train_input, train_output, train_label = None, None, None

    for ep in range(epoch):
        print("{}-{} | Epoch {}".format(model_name, version, ep))
        train_iter, dev_iter = iter(train_loader), iter(dev_loader)
        # TRAIN
        for _ in tqdm(range(train_num_mini_batches)):
            train_input, train_label = train_iter.next()
            train_input, train_label = train_input.to(CUDA_DEVICE), train_label.to(CUDA_DEVICE)
            # optimizer.zero_grad()
            train_output = net(train_input)
            train_loss = compute_loss(train_output, train_label)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        # DEV
        with torch.no_grad():
            for _ in range(dev_num_mini_batches):
                dev_input, dev_label = dev_iter.next()
                dev_input, dev_label = dev_input.to(CUDA_DEVICE), dev_label.to(CUDA_DEVICE)
                dev_output = net(dev_input)
                dev_loss = compute_loss(dev_output, dev_label)
                running_dev_loss += dev_loss.item()

        # record loss values after each epoch
        cur_train_loss = running_train_loss / train_num_mini_batches
        cur_dev_loss = running_dev_loss / dev_num_mini_batches
        print("train loss = {:.4} | val loss = {:.4}".format(cur_train_loss, cur_dev_loss))
        tb.add_scalar('loss/train', cur_train_loss, ep)
        tb.add_scalar('loss/dev', cur_dev_loss, ep)
        if ep % 5 == 0:
            tensorboard_vis(tb, ep, mode='train', input_=train_input, output=train_output, label=train_label)
            tensorboard_vis(tb, ep, mode='dev', input_=dev_input, output=dev_output, label=dev_label)
        running_train_loss, running_dev_loss = 0.0, 0.0

    print("finished training")
    save_network_weights(net, ep="{}_FINAL".format(epoch))
    return


def parse_args():
    parser = argparse.ArgumentParser(description='Specify target GPU, else the one defined in config.py will be used.')
    parser.add_argument('--gpu', type=int, help='cuda:$')
    args = parser.parse_args()
    if args.gpu is not None:
        CUDA_DEVICE = "cuda:{}".format(args.gpu)
    else:
        CUDA_DEVICE = "cpu".format(args.gpu)
    return CUDA_DEVICE


def main():
    global version, model_name, CUDA_DEVICE
    CUDA_DEVICE = parse_args()
    model_name, version = "unet16", "v0.8.0"
    param_to_load = None
    tb = SummaryWriter('./runs/' + model_name + '-' + version)
    # net = UNet11(pretrained=True)
    net = UNet16(pretrained=True)
    train_dev(net, tb, load_weights=False, pre_trained_params_path=None)
    tb.close()


if __name__ == "__main__":
    main()
