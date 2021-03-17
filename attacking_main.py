import torch.utils.data as data
import base_model
import pdb
import argparse
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import Image
from data_process import *
from experiment_operator import Experiment_Operator

if __name__ == "__main__":
    experiment_settings = {"dataset": "cifar10", "batch_size": 100, "lr": 0.1}

    # parser = argparse.ArgumentParser(description = "adversarial learning")
    # parser.add_argument()

    train_dset = datasets.CIFAR10(root = './datasets', train = True, download = False)
    test_dset = datasets.CIFAR10(root = './datasets', train = False, download = False)

    any_picture = test_dset.data[26] # 26,
    any_picture_target = test_dset.targets[26]
    print(any_picture_target)
    print(test_dset.classes[any_picture_target])


    visualize(any_picture)

    my_model = base_model.myResNet18(pretrained = False, input_channels=3, first_padding=1, num_classes = 10)

    my_morm = Normalize()

    PIL_any_picture = Image.fromarray(any_picture.astype(np.uint8))
    transform_image = test_transform()(PIL_any_picture)[None, :, :, :]

    my_datasets = {"train": My_Dataset(train_dset, transform=train_transform(image_size=train_dset.data.shape[1])),
                   "test": My_Dataset(test_dset, transform=test_transform())}
    dset_loaders = {"train": data.DataLoader(dataset=my_datasets["train"], batch_size=experiment_settings["batch_size"], shuffle=True, num_workers=2),
                    "test": data.DataLoader(dataset=my_datasets["test"], batch_size=experiment_settings["batch_size"], shuffle=False, num_workers=2)}
    experiment_op = Experiment_Operator(datasets_loaders=dset_loaders,
                                        norm=my_morm,
                                        model=my_model,
                                        batch_size=experiment_settings["batch_size"],
                                        milestones=None,
                                        lr=experiment_settings["lr"],
                                        scale=1.0,
                                        is_BN=True,
                                        is_gpu=True)

    experiment_op.load_model(path = "model_weights/cifar10-imagenet-resnet18.pth")
    a, b, c, d = experiment_op.test()
    print(c, d)


    pred = experiment_op.model(experiment_op.norm(transform_image.cuda()))
    #
    minus_ln_p = experiment_op.criterion(pred, torch.LongTensor([any_picture_target]).cuda())
    print(np.exp(-minus_ln_p.item()))
    print(pred.max(dim = 1))

    print("Start attacking...")
    delta = experiment_op.sample_attack_train(transform_image.cuda(), torch.LongTensor([any_picture_target]).cuda())
    visualize((transform_image + delta.detach().cpu())[0].numpy().transpose(1, 2, 0))
    visualize((50 * delta.detach().cpu().numpy() + 0.5)[0].transpose(1,2,0))
