import torch
import torch.utils.data as data
import base_model
import pdb
from torchvision import transforms, datasets
from PIL import Image
from data_process import My_Dataset, Normalize, my_transform

if __name__ == "__main__":
    experiment_settings = {"dataset": "cifar10", }
    train_dset = datasets.MNIST(root='./datasets', train = True, download = True)
    test_dset = datasets.MNIST(root='./datasets', train = False, download = True)
    pdb.set_trace()

    if experiment_settings["dataset"] == "cifar10":
        train_dset = datasets.CIFAR10(root = './datasets', train = True, download = False)
        test_dset = datasets.CIFAR10(root = './datasets', train = False, download = False)

    elif experiment_settings["dataset"] == "cifar10":
        pass

    datasets = {"train": My_Dataset(train_dset, transform = my_transform),
                "test": My_Dataset(test_dset, transform = my_transform)}

    dset_loaders = {"train": data.DataLoader(dataset = train_dset, batch_size=32, shuffle=True, num_workers=4),
                    "test": data.DataLoader(dataset = test_dset, batch_size=32, shuffle=True, num_workers=4)}
    # print(train_dset.data.shape)
    # print(train_dset.data[0].shape)
    # # print(train_dset.data[0].transpose(1, 2, 0).shape)
    # tr_dset = my_transform(224)(Image.fromarray(train_dset.data[0]))
    # print(tr_dset.shape)
    # pdb.set_trace()

