import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pdb

class Normalize(nn.Module):

    # Using mean and std calculated from ImageNet as default
    # ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # tensor([0.4914, 0.4822, 0.4465], dtype=torch.float64)
    # tensor([0.2470, 0.2435, 0.2616], dtype=torch.float64)

    def __init__(self, mean = torch.Tensor([0.485, 0.456, 0.406]), std = torch.Tensor([0.229, 0.224, 0.225])):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        '''
        :param x: tensor of size [B, C, H, W]
        :return: normalized tensor of x
        '''
        x = (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]
        return x

def my_transform(new_size = 224):
    return transforms.Compose([
        transforms.Resize(new_size), # a time-consuming operation, should be avoided as possible.
        transforms.ToTensor(),
        ])

def train_transform(image_size = 32):
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

def test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def channel_duplicate():
    pass


class My_Dataset(Dataset):
    def __init__(self, Dataset, transform = my_transform):

        if isinstance(Dataset.data, np.ndarray):
            self.data = Dataset.data
        else: # torch.Tensor # MNIST
            # self.data = np.expand_dims(Dataset.data.numpy(), axis = 3).repeat(3, axis = 3)
            self.data = Dataset.data.numpy()

        # if len(self.data.shape) < 4:
        #     self.data = self.data[:, :, :, None]

        self.targets = Dataset.targets
        # we don't need mean & std here because we shouldn't normalize data here, see https://adversarial-ml-tutorial.org/introduction/
        # self.mean = torch.Tensor([0.485, 0.456, 0.406])
        # self.std = torch.Tensor([0.229, 0.224, 0.225])
        self.dict = dict(zip(Dataset.class_to_idx.values(), Dataset.class_to_idx.keys())) # from class_to_idx to idx_to_class
        self.transform = transform
        self.num_images = self.data.shape[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        ndarray_image = self.data[idx]
        PIL_image = Image.fromarray(ndarray_image.astype(np.uint8))
        transform_image = self.transform(PIL_image)
        target_class = self.targets[idx]

        return transform_image, target_class

    def compute(self):
        '''
        :param dataset: ndarray of shape [B, W, H, C]
        :return: mean & std tensor of three channels
        '''

        self.mean = torch.mean(torch.from_numpy(self.data.astype(np.float)), dim = [0, 1, 2])
        self.std = torch.std(torch.from_numpy(self.data.astype(np.float)), dim = [0, 1, 2], unbiased = True) # using Bessel’s correction to do the unbiased estimation
        if len(self.data.shape) < 4:
            self.mean = torch.Tensor([self.mean.item()])
            self.std = torch.Tensor([self.std.item()])
        return self.mean / 255.0, self.std / 255.0

def visualize(data_array):
    plt.imshow(data_array)
    plt.show()

if __name__ == "__main__":
    a = torch.Tensor([2, 3, 4, 5])
    print(torch.Tensor(5))

    b = torch.Tensor([[[1, 2, 3, 4], [2, 3, 4, 5]]])
    print(b - torch.Tensor([2]).type_as(b)[:, None, None])
    print(b[0])
    # print(np.expand_dims(b, axis = 2).repeat(3, axis = 0))
    # print(np.newaxis)
    # print(len(b[:, None, :, None].shape))
    # print(len(b.shape))
    # print(isinstance(a, np.ndarray))
    # print(torch.from_numpy(a))
