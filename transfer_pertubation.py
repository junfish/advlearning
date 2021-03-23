import numpy as np
import torch.utils.data as data
import base_model
import pdb
import argparse
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image
from data_process import *
from experiment_operator import Experiment_Operator
perturbation = np.load("pertubation.npy")
# print(perturbation)



train_dset = datasets.CIFAR10(root = './datasets', train = True, download = False)
test_dset = datasets.CIFAR10(root = './datasets', train = False, download = False)

any_picture = test_dset.data[1034]
any_picture_target = test_dset.targets[1034]
PIL_any_picture = Image.fromarray(any_picture.astype(np.uint8))
transform_image = test_transform()(PIL_any_picture)[None, :, :, :]
print("The chosen picture's idx and class are as below:")
print(any_picture_target)
print(test_dset.classes[any_picture_target])
print("Visualize this picture:")
visualize(np.squeeze(transform_image.numpy() + perturbation).transpose(1, 2, 0))
visualize(any_picture)


num_channels = 3
num_paddings = 1
my_model = base_model.myResNet18(pretrained=False, input_channels=num_channels, first_padding=num_paddings, num_classes=10)


my_datasets = {"train": My_Dataset(train_dset, transform=train_transform(image_size=train_dset.data.shape[1])),
                "test": My_Dataset(test_dset, transform=test_transform())}
dset_loaders = {"train": data.DataLoader(dataset=my_datasets["train"], batch_size=100, shuffle=True, num_workers=2),
                "test": data.DataLoader(dataset=my_datasets["test"], batch_size=100, shuffle=False, num_workers=2)}
datasets_mean, datasets_std = my_datasets["train"].compute()
my_morm = Normalize(mean=datasets_mean, std=datasets_std)
experiment_op = Experiment_Operator(datasets_loaders=dset_loaders,
                                    norm=my_morm,
                                    model=my_model,
                                    batch_size=100,
                                    milestones=None,
                                    lr=0.1,
                                    scale=1.0,
                                    is_BN=True,
                                    is_gpu=True)
prediction = experiment_op.model(experiment_op.norm(torch.from_numpy(transform_image.numpy() + perturbation).cuda()))
print(prediction)
print(nn.Softmax(dim=1)(prediction)[0, prediction.max(dim=1)[1].item()].item())