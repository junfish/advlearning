import torch
import torch.utils.data as data
import base_model
import pdb
from torchvision import transforms, datasets
from PIL import Image
from data_process import My_Dataset, Normalize, my_transform

if __name__ == "__main__":
    experiment_settings = {"dataset": "cifar10", "batch_size": 100}

    train_dset = datasets.CIFAR10(root='./datasets', train=True, download=False)
    test_dset = datasets.CIFAR10(root='./datasets', train=False, download=False)
    if experiment_settings["dataset"] == "mnist":
        train_dset = datasets.MNIST(root='./datasets', train = True, download = False)
        test_dset = datasets.MNIST(root='./datasets', train = False, download = False)

    my_model = base_model.myResNet18(pretrained = True, num_classes = 10)


    xdatasets = {"train": My_Dataset(train_dset, transform = my_transform(new_size = 224)),
                "test": My_Dataset(test_dset, transform = my_transform(new_size = 224))}

    dset_loaders = {"train": data.DataLoader(dataset = xdatasets["train"], batch_size=32, shuffle=True, num_workers=4),
                    "test": data.DataLoader(dataset = xdatasets["test"], batch_size=32, shuffle=True, num_workers=4)}

    pdb.set_trace()
    # print(train_dset.data.shape)
    # print(train_dset.data[0].shape)
    # # print(train_dset.data[0].transpose(1, 2, 0).shape)
    # tr_dset = my_transform(224)(Image.fromarray(train_dset.data[0]))
    # print(tr_dset.shape)
    # pdb.set_trace()

