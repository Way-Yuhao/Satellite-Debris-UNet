from torchvision.datasets.vision import VisionDataset
import torch
import torchvision.transforms as T
import cv2
from tqdm import tqdm
import os
import os.path as p
import numpy as np
from natsort import natsorted
from config import CROP_WIDTH, CROP_HEIGHT
from matplotlib import pyplot as plt

def cv_loader(path):
    """
    loads .hdr file via cv2, then converts color to rgb
    :param path: path to image file
    :return: img ndarray
    """
    img = cv2.imread(path, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype("float32")
    return img


def main():
    path = "/mnt/data1/yl241/datasets/Inria_Aerial/AerialImageDataset/train/images/"
    img = cv_loader(p.join(path, 'austin1.tif'))

    input_transforms = T.Compose([
        T.ToTensor(),
        T.RandomCrop(CROP_HEIGHT, CROP_WIDTH),
        T.ConvertImageDtype(torch.float),
        lambda x: x / 255.
        # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    proc_img = input_transforms(img)
    print(proc_img)


if __name__ == '__main__':
    main()
