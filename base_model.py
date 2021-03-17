import torch
from torch import nn
import torchvision
from torchsummary import summary
import argparse

class two_layer_CNN(nn.Module):

    def __init__(self, input_channels = 3, first_padding = 0, num_classes = 10):
        super(two_layer_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = 6, kernel_size = 5, stride = 1, padding = first_padding)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)

        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


class myResNet18(nn.Module):
    def __init__(self, input_channels = 3, first_padding = 1, pretrained = False, num_classes = 10):
        super(myResNet18, self).__init__()
        raw_resnet18 = torchvision.models.resnet18(pretrained = pretrained)
        self.conv1 = nn.Conv2d(input_channels, raw_resnet18.conv1.out_channels, 3, 1, first_padding, bias = False)
        # self.conv1 = raw_resnet18.conv1
        self.bn1 = raw_resnet18.bn1
        self.relu = raw_resnet18.relu
        self.maxpool = raw_resnet18.maxpool
        self.layer1 = raw_resnet18.layer1
        self.layer2 = raw_resnet18.layer2
        self.layer3 = raw_resnet18.layer3
        self.layer4 = raw_resnet18.layer4
        self.avgpool = raw_resnet18.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(raw_resnet18.inplanes, num_classes)

    def forward(self, x):
        # input [B, input_channels, 224, 224] or [B, input_channels, 32, 32]
        x = self.conv1(x) # [B, 64, 112, 112] or [B, 64, 32, 32] (delete max-pooling)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, 56, 56]

        x = self.layer1(x) # [B, 64, 56, 56]
        x = self.layer2(x) # [B, 128, 28, 28] # size changes inside the block.
        x = self.layer3(x) # [B, 256, 14, 14]
        x = self.layer4(x) # [B, 512, 7, 7]

        x = self.avgpool(x) # [B, 512, 1, 1]
        x = self.flatten(x) # start_dim = 1, end_dim = -1, [B, 512]
        x = self.fc(x) # [B, num_classes]

        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("echo", help = "echo the string you use here!")
    parser.add_argument("square", help = "print the square value.", type = float)
    parser.add_argument("--verbosity", help = "increase the verbosity.")
    parser.add_argument("-g", "--use_gpu", help = "use gpu or not.", action = "store_true")
    parser.add_argument("--dataset", help = "choose a dataset.", type = int, choices = [0, 1, 2])
    args = parser.parse_args()
    print(args)
    print(args.echo)
    print(args.square ** 2)
    if args.verbosity:
        print("verbosity turned on")
    else:
        print("verbosity is not turned on")
    if args.use_gpu:
        print("we are using gpu!")
    else:
        print("we are not using gpu!")
    if args.dataset == 0:
        print("cifar10")
    elif args.dataset == 1:
        print("cifar100")
    elif args.dataset == 2:
        print("imagenet")
    # my_resnet = myResNet18(pretrained = False, num_classes = 10)
    # my_two_layer_CNN = two_layer_CNN(input_channels = 3, first_padding = 0, num_classes = 10)
    # # summary(my_resnet, input_size = (3, 32, 32))
    # summary(my_two_layer_CNN, input_size = (3, 32, 32))