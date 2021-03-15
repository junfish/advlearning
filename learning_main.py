import torch.utils.data as data
import base_model
import pdb

from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image
from data_process import *
from experiment_operator import Experiment_Operator

if __name__ == "__main__":
    experiment_settings = {"dataset": "cifar10", "batch_size": 100, "normalize_by_self": True, "lr": 0.1}

    train_dset = datasets.CIFAR10(root='./datasets', train=True, download=False)
    test_dset = datasets.CIFAR10(root='./datasets', train=False, download=False)
    if experiment_settings["dataset"] == "mnist":
        train_dset = datasets.MNIST(root='./datasets', train = True, download = False)
        test_dset = datasets.MNIST(root='./datasets', train = False, download = False)

    my_model = base_model.myResNet18(pretrained = False, num_classes = 10)

    my_datasets = {"train": My_Dataset(train_dset, transform = train_transform()),
                   "test": My_Dataset(test_dset, transform = test_transform())}

    datasets_mean, datasets_std = my_datasets["train"].compute()
    print(datasets_mean)
    print(datasets_std)

    my_morm = Normalize()

    dset_loaders = {"train": data.DataLoader(dataset = my_datasets["train"], batch_size = experiment_settings["batch_size"], shuffle=True, num_workers=2),
                    "test": data.DataLoader(dataset = my_datasets["test"], batch_size = experiment_settings["batch_size"], shuffle=True, num_workers=2)}

    experiment_op = Experiment_Operator(datasets_loaders = dset_loaders,
                                        norm = my_morm,
                                        model = my_model,
                                        batch_size = experiment_settings["batch_size"],
                                        lr = experiment_settings["lr"],
                                        scale = 1.0,
                                        is_BN = True,
                                        is_gpu = True)
    experiment_op.train(iterations = 400)
    experiment_op.save_model(path = "model_weights/cifar10-resnet18.pth")
