import numpy as np
import torch
from torch.utils.data import Dataset


class base_loader(Dataset):
    def __init__(self, samples, labels, domains):
        self.samples = samples
        self.labels = labels
        self.domains = domains

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        return sample, target, domain

    def __len__(self):
        return len(self.samples)


class HARDataset(Dataset):
    def __init__(self, data_tensor, target_tensor=None, domain_tensor=None, transforms=None):
        if target_tensor is not None:
            assert len(data_tensor) == len(target_tensor)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.domain_tensor = domain_tensor
        if transforms is None:
            transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]

        self.transforms = transforms

    def __getitem__(self, index):
        data = self.data_tensor[index]
        for transform in self.transforms:
            data = transform(data)
        target = None
        if self.target_tensor is None:
            target = torch.tensor([])
        else:
            target = self.target_tensor[index]
        domain = None
        if self.domain_tensor is None:
            domain = torch.tensor([])
        else:
            domain = self.domain_tensor[index]
        # Dataset __getitem__不能返回None值
        return data, target, domain

    def __len__(self):
        return len(self.data_tensor)
