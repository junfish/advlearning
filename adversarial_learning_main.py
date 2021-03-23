import torch.utils.data as data
import base_model
import pdb
import argparse
from argparse import RawTextHelpFormatter
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image
from data_process import *
from experiment_operator import Experiment_Operator

# class SmartFormatter(argparse.HelpFormatter):
#     def _split_lines(self, text, width):
#         if text.startswith('R|'):
#             return text[2:].splitlines()
#         # this is the RawTextHelpFormatter._split_lines
#         return argparse.HelpFormatter._split_lines(self, text, width)
#
# class My_Formatter(RawDescriptionHelpFormatter, SmartFormatter): pass

if __name__ == "__main__":
    # experiment_settings = {"dataset": "mnist", "batch_size": 100, "normalize_by_self": True, "lr": 0.001}
    parser = argparse.ArgumentParser(description = "This is the main training program.\n"
                                                   "Firstly, You can specify the dataset and network for the learning process.\n"
                                                   "Secondly, You can also set hyper-parameters for your model training.\n"
                                                   "More details can be seen from the introduction of arguments below .",
                                     formatter_class = RawTextHelpFormatter)

    parser.add_argument("model", type = int, choices = [0, 1], #default = 1,
                        help = "specify a deep learning model (1 as default)\n"
                             "0: LeNet (a CNN with two conv layers)\n"
                             "1: ResNet18 (change the first conv layer a little bit)")

    parser.add_argument("dataset", type = int, choices = [0, 1], #default = 1,
                        help = "specify a dataset to learn (1 as default)\n"
                               "0: mnist\n"
                               "1: cifar10")

    parser.add_argument("-l", "--lr", type = float, default = 0.1, help = "give an initial learning rate to our model")
    parser.add_argument("-n", "--norm", type = int, default = 0, help = "choose the norm for normalization\n"
                                                                        "0: imagenet mean and std\n"
                                                                        "1: compute own mean and std")
    parser.add_argument("-b", "--batch_size", type = int, default = 100, help = "set a batch size for the training and testing process")
    parser.add_argument("-i", "--iterations", type = int, default = 100, help = "the total number of epoch for the training")
    parser.add_argument("-s", "--schedule_intervals", nargs = "+", type = int,
                        help = "intervals to degrade learning rate\n"
                               "e.g., recommendation: start at lr = 0.1, intervals = [135, 230, 300], divided by 10, for cifar10 & resnet18")

    # parser.add_argument("")
    args = parser.parse_args()
    print(args)

    if args.dataset == 0:
        print("Loading mnist to the memory...")
        train_dset = datasets.MNIST(root = './datasets', train = True, download = False)
        test_dset = datasets.MNIST(root = './datasets', train = False, download = False)
        num_channels = 1
        num_paddings = 3

    elif args.dataset == 1:
        print("Loading cifar10 to the memory.")
        train_dset = datasets.CIFAR10(root = './datasets', train = True, download = False)
        test_dset = datasets.CIFAR10(root = './datasets', train = False, download = False)
        num_channels = 3
        num_paddings = 1

    if args.model == 0:
        print("Initializing 2 Layer CNN...")
        my_model = base_model.two_layer_CNN(input_channels = num_channels, first_padding = num_paddings, num_classes = 10)
    elif args.model == 1:
        print("Initializing ResNet18...")
        my_model = base_model.myResNet18(pretrained=False, input_channels=num_channels, first_padding=num_paddings, num_classes=10)

    my_datasets = {"train": My_Dataset(train_dset, transform = train_transform(image_size = train_dset.data.shape[1])),
                   "test": My_Dataset(test_dset, transform = test_transform())}

    if args.norm == 0:
        my_morm = Normalize()
    elif args.norm == 1:
        datasets_mean, datasets_std = my_datasets["train"].compute()
        my_morm = Normalize(mean=datasets_mean, std=datasets_std)
    # datasets_mean, datasets_std = my_datasets["test"].compute()


    # my_morm = Normalize()

    dset_loaders = {"train": data.DataLoader(dataset = my_datasets["train"], batch_size = args.batch_size, shuffle=True, num_workers = 2),
                    "test": data.DataLoader(dataset = my_datasets["test"], batch_size = args.batch_size, shuffle=False, num_workers = 2)}

    experiment_op = Experiment_Operator(datasets_loaders = dset_loaders,
                                        norm = my_morm,
                                        model = my_model,
                                        batch_size = args.batch_size,
                                        milestones = args.schedule_intervals,
                                        lr = args.lr,
                                        scale = 1.0,
                                        is_BN = True,
                                        is_gpu = True)
    print("Start training...")
    # experiment_op.robust_train(iterations = args.iterations)

    experiment_op.robust_train(iterations=args.iterations)
    experiment_op.save_model(path ="model_weights/robust-mnist-resnet18.pth")
