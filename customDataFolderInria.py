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


class ImageFolderInria(VisionDataset):

    def __init__(self, dataset_dir):
        # super().__init__(root)
        self.input_dir = p.join(dataset_dir, 'images')
        self.label_dir = p.join(dataset_dir, 'gt')
        self.inputs = natsorted(os.listdir(self.input_dir))
        self.labels = natsorted(os.listdir(self.label_dir))
        assert(len(self.inputs) == len(self.labels),
               "ERROR: # of inputs ({})and # of labels ({})do not match.".format(len(self.inputs), len(self.labels)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = cv_loader(p.join(self.input_dir, self.inputs[idx]))
        label_sample = cv_loader(p.join(self.label_dir, self.labels[idx]))

        input_sample, label_sample = self.normalize(input_sample, label_sample)
        input_sample, label_sample = self.augment(input_sample, label_sample)

        return input_sample, label_sample

    def augment(self, input_, label):
        # input_, label = self.random_crop(input_, label)
        # input_, label = self.random_horizontal_flip(input_, label)
        # input_, label = self.random_rotation(input_, label)

        transforms = T.Compose([
            T.ToTensor(),
            T.RandomCrop((CROP_HEIGHT, CROP_WIDTH)),
            T.RandomHorizontalFlip(p=0.5)
        ])
        input_, label = transforms(input_), transforms(label)
        return input_, label

    def normalize(self, input_, label):
        input_ = input_ / 255.
        return input_, label

    def rotate(self, input_, label):
        raise NotImplementedError()  # TODO

    def random_crop(self, input_, label):
        max_h = input_.shape[1] - CROP_HEIGHT
        max_w = input_.shape[2] - CROP_WIDTH

        h = np.random.randint(0, max_h / 2) * 2  # random even numbers
        w = np.random.randint(0, max_w / 2) * 2
        input_crop = input_[:, h: h + CROP_HEIGHT, w: w + CROP_WIDTH]
        label_crop = label[:, h: h + CROP_HEIGHT, w: w + CROP_WIDTH]
        return input_crop, label_crop

    def random_horizontal_flip(self, input_, label, p=.5):
        """
        applies a random horizontal flip
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :param p: probability of applying the flip
        :return: flipped or original images
        """
        x = np.random.rand()
        if x < p:
            input_ = torch.flip(input_, (2,))
            label = torch.flip(label, (2,))

        return input_, label

    def random_rotation(self, input_, label, p=.5):
        """
        applies a random 180 degree rotation
        :param input_: CMOS input
        :param spad: SPAD input
        :param target: ground truth
        :param p: probability of applying the rotation
        :return: rotated or original images
        """
        x = np.random.rand()  # probability of rotation
        if x < p:
            input_ = torch.rot90(input_, 2, [1, 2])
            label = torch.rot90(label, 2, [1, 2])

        return input_, label
