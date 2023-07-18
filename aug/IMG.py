import os

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT

from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import PIL
import cv2


mean = {
    'mnist': (0.1307,),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

std = {
    'mnist': (0.3081,),
    'cifar10': (0.2470, 0.2435, 0.2616)
}


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * \
                    np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(
                sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class CenterCropAndResize(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, proportion, size):
        self.proportion = proportion
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped and image.
        """
        w, h = (np.array(img.size) * self.proportion).astype(int)
        img = FT.resize(
            FT.center_crop(img, [int(h), int(w)]),
            [self.size, self.size],
            interpolation=PIL.Image.BICUBIC
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(proportion={0}, size={1})'.format(self.proportion, self.size)


class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort
