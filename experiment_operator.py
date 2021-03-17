import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import base_model
import pdb
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image
from data_process import My_Dataset, Normalize, my_transform

class Experiment_Operator(object):

    def __init__(self, datasets_loaders, norm, model, batch_size, milestones, target_parameters = None, lr = 1e-1, scale = 1.0, is_BN = True, is_gpu = True):
        '''
        :param datasets_loaders:
        :param norm:
        :param model:
        :param batch_size:
        :param milestones:
        :param target_parameters:
        :param lr:
        :param scale:
        :param is_BN:
        :param is_gpu:
        '''

        if milestones is None:
            milestones = [135, 230, 300]
        self.train_loaders = datasets_loaders["train"]
        self.test_loaders = datasets_loaders["test"]
        self.batch_size = batch_size
        self.milestones = milestones
        self.lr = lr
        self.scale = scale
        self.is_BN = is_BN
        self.is_gpu = is_gpu

        if self.is_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.norm = norm.cuda()
            self.model = model.cuda()

        else:
            self.norm = norm
            self.model = model
        self.criterion = nn.CrossEntropyLoss()

        if target_parameters:
            self.optimizer = optim.SGD(target_parameters, lr = self.scale * self.lr, momentum = 0.9, weight_decay = 1e-5)
            self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.1)
            self.handcraft_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.milestones, gamma = 0.1)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum = 0.9, weight_decay = 1e-4)
            self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.1)
            self.handcraft_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.milestones, gamma = 0.1)

    def train(self, iterations = 100):
        '''
        :param iterations: training iterations
        :return: nothing
        '''
        self.model.train()
        for epoch in range(iterations):
            for (i, data) in enumerate(self.train_loaders):
                images, images_target = data
                if self.is_gpu:
                    inputs = Variable(images).cuda()
                    labels = Variable(images_target).cuda()
                else:
                    inputs = Variable(images)
                    labels = Variable(images_target)

                self.optimizer.zero_grad()
                outputs = self.model(self.norm(inputs))
                batch_loss = self.criterion(outputs, labels) # outputs is one-hot code, and labels is the number of class.
                batch_loss.backward()
                self.optimizer.step()
            self.handcraft_lr_scheduler.step()
            training_loss, testing_loss, training_acc, testing_acc = self.test()
            print("Learning Rate = %.8f" % self.handcraft_lr_scheduler.get_last_lr()[0])
            print("Epoch %03d: training_accuracy = %.4f, testing_accuracy = %.4f" % (epoch + 1, training_acc, testing_acc))
            print("           training_loss = %.6f, testing_loss = %.6f" % (training_loss, testing_loss))
        print("Finished Training after %03d iterations" % iterations)

    def test(self):
        if self.is_BN:
            self.model.eval()
        acc_count = 0
        training_loss = 0.0
        testing_loss = 0.0
        for (i, data) in enumerate(self.train_loaders):
            images, images_target = data
            if self.is_gpu:
                inputs = Variable(images).cuda()
                labels = Variable(images_target).cuda()
            else:
                inputs = Variable(images)
                labels = Variable(images_target)
            outputs = self.model(self.norm(inputs))
            training_loss += self.criterion(outputs, labels).item()
            output_labels = torch.argmax(outputs, dim = 1)
            # print(labels)
            # print(output_labels)
            for (y, y_bar) in zip(output_labels, labels):
                if y == y_bar:
                    acc_count += 1
        training_acc = acc_count / len(self.train_loaders) / self.batch_size
        training_loss /= (len(self.train_loaders) * self.batch_size)
        acc_count = 0
        for (i, data) in enumerate(self.test_loaders):
            images, images_target = data
            if self.is_gpu:
                inputs = Variable(images).cuda()
                labels = Variable(images_target).cuda()
            else:
                inputs = Variable(images)
                labels = Variable(images_target)
            outputs = self.model(self.norm(inputs))
            testing_loss += self.criterion(outputs, labels).item()
            output_labels = torch.argmax(outputs, dim = 1)
            # print(labels)
            # print(output_labels)
            for (y, y_bar) in zip(output_labels, labels):
                if y == y_bar:
                    acc_count += 1
        testing_acc = acc_count / len(self.test_loaders) / self.batch_size
        testing_loss /= (len(self.test_loaders) * self.batch_size)
        return training_loss, testing_loss, training_acc, testing_acc

    def sample_attack_train(self, sample, sample_target, epsilon = 2.0 / 255, iterations = 50):
        '''
        :param sample:
        :param sample_target:
        :param epsilon:
        :param iterations:
        :return:
        '''
        delta = torch.zeros_like(sample, requires_grad = True).cuda()
        print(delta.shape)
        opt = optim.SGD([delta], lr = 0.1)
        # self.model.eval()
        for epoch in range(iterations):

            prediction = self.model(self.norm(sample + delta))
            loss = -nn.CrossEntropyLoss()(prediction, sample_target)
            if (epoch + 1) % 5 == 0:
                # print(delta[0][0])
                print(epoch + 1, loss.item())
                # print("dfgh")

            opt.zero_grad()
            loss.backward()
            opt.step()
            delta.data.clamp_(-epsilon, epsilon)
        print("True class probability:", nn.Softmax(dim=1)(prediction)[0, sample_target].item())
        return delta


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def compute_gradient(self, x, y):
        sgd_optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9)
        sgd_optimizer.zero_grad()
        input = Variable(x, requires_grad = True).cuda()
        label = Variable(y).cuda()
        output = self.model(self.norm(input))
        loss = self.criterion(output, label)
        loss.backward()
        sgd_optimizer.step()
        return input.grad

if __name__ == "__main__":
    experiment_settings = {"dataset": "cifar10", "batch_size": 100, "lr": 0.1}

    train_dset = datasets.CIFAR10(root='./datasets', train=True, download=False)
    test_dset = datasets.CIFAR10(root='./datasets', train=False, download=False)
    if experiment_settings["dataset"] == "mnist":
        train_dset = datasets.MNIST(root='./datasets', train = True, download = False)
        test_dset = datasets.MNIST(root='./datasets', train = False, download = False)

    my_model = base_model.myResNet18(pretrained = True, num_classes = 10)

    my_datasets = {"train": My_Dataset(train_dset, transform = my_transform(new_size = 224)),
                "test": My_Dataset(test_dset, transform = my_transform(new_size = 224))}

    my_morm = Normalize()

    dset_loaders = {"train": data.DataLoader(dataset = my_datasets["train"], batch_size = experiment_settings["batch_size"], shuffle=True, num_workers=2),
                    "test": data.DataLoader(dataset = my_datasets["test"], batch_size = experiment_settings["batch_size"], shuffle=True, num_workers=2)}

    experiment_op = Experiment_Operator(datasets_loaders = dset_loaders,
                                        norm = my_morm,
                                        model = my_model,
                                        batch_size = experiment_settings["batch_size"],
                                        milestones = [135, 230, 300],
                                        lr = experiment_settings["lr"],
                                        scale = 1.0,
                                        is_BN = True,
                                        is_gpu = True)
    experiment_op.train(iterations = 100)
    experiment_op.save_model(path = "model_weights/cifar10-resnet18.pth")

