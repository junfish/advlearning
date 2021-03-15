import torch.utils.data as data
import base_model
import pdb
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image
from data_process import *
from experiment_operator import Experiment_Operator

if __name__ == "__main__":
    experiment_settings = {"dataset": "cifar10", "batch_size": 100, "lr": 0.1}
    train_dset = datasets.CIFAR10(root = './datasets', train = True, download = False)
    test_dset = datasets.CIFAR10(root = './datasets', train = False, download = False)

    any_picture = test_dset.data[0] # cat - 3
    print(any_picture[:, :, 0])


    visualize(any_picture)

    # my_model = base_model.myResNet18(pretrained = True, num_classes = 10)
    #
    # my_morm = Normalize()
    #
    # experiment_op = Experiment_Operator(datasets_loaders = None,
    #                                     norm = my_morm,
    #                                     model = my_model,
    #                                     batch_size = None,
    #                                     lr = experiment_settings["lr"],
    #                                     scale = 1.0,
    #                                     is_BN = True,
    #                                     is_gpu = True)
    #
    # experiment_op.load_model(path = "model_weights/cifar10-resnet18.pth")

