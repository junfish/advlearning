import torch
from torch import nn
import torchvision
from torchsummary import summary

class myResNet18(nn.Module):
    def __init__(self, pretrained = True, input_channels = 3, num_classes = 10):
        super(myResNet18, self).__init__()
        raw_resnet18 = torchvision.models.resnet18(pretrained = pretrained)
        self.conv1 = nn.Conv2d(input_channels, raw_resnet18.inplanes, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = raw_resnet18.bn1
        self.relu = raw_resnet18.relu
        self.maxpool = raw_resnet18.maxpool
        self.layer1 = raw_resnet18.layer1
        self.layer2 = raw_resnet18.layer2
        self.layer3 = raw_resnet18.layer3
        self.layer4 = raw_resnet18.layer4
        self.avgpool = raw_resnet18.avgpool
        self.fc = nn.Linear(raw_resnet18.inplanes, num_classes)

    def forward(self, x):
        # input [B, input_channels, 224, 224]
        x = self.conv1(x) # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [B, 64, 56, 56]

        x = self.layer1(x) # [B, 64, 56, 56]
        x = self.layer2(x) # [B, 128, 28, 28] # size changes inside the block.
        x = self.layer3(x) # [B, 256, 14, 14]
        x = self.layer4(x) # [B, 512, 7, 7]

        x = self.avgpool(x) # [B, 512, 1, 1]
        x = torch.flatten(x, 1) # start_dim = 1, end_dim = -1, [B, 512]
        x = self.fc(x) # [B, num_classes]

        return x

if __name__ == "__main__":
    my_model = myResNet18(pretrained = False, num_classes = 10)
    summary(my_model, input_size = (3, 224, 224))