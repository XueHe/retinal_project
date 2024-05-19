import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import cv2
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=1024, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        """Reads an image from a file, preprocesses it, and returns."""
        # Read image and label paths
        image_path = self.image_paths[index]
        filename = os.path.split(image_path)[-1].split('.')[0]
        GT_path = os.path.join(self.GT_paths, filename + '.png')

        # Load image and label
        image = Image.open(image_path)  # Convert to grayscale
        GT = Image.open(GT_path) 
        # Convert images to tensors
        image_tensor = to_tensor(image)
        GT_tensor = to_tensor(GT)

        return image_tensor, GT_tensor

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    dataset = ImageFolder(root='./dataset/train/', image_size=1024, augmentation_prob=0.4)
    print("数据个数：", len(dataset))
    train_loader = get_loader(image_path='./dataset/preprocessed_train/',
                              image_size='./dataset/preprocessed_valid/',
                              batch_size=1,
                              num_workers=6,
                              mode='train',
                              augmentation_prob=0.4)
    for image, label in train_loader:
        print(image.shape)