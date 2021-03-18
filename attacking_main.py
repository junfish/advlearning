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
    # experiment_settings = {"dataset": "cifar10", "batch_size": 100, "lr": 0.1}

    parser = argparse.ArgumentParser(description = "attack learning to generate an adversarial sample")
    parser.add_argument("model", type=int, choices = [0, 1], help = "same to learning_main.py")
    parser.add_argument("dataset", type = int, choices = [0, 1], help = "same to learning_main.py")
    parser.add_argument("-l", "--lr", type=float, default=0.1, help="give an initial learning rate to our model")
    parser.add_argument("-n", "--norm", type=int, default=0, help="choose the norm for normalization")
    parser.add_argument("-b", "--batch_size", type = int, default = 100, help = "set a batch size for the data loader")
    parser.add_argument("-s", "--sample", type = int, default = 26, help=
                        "give a number to choose a picture from the testing dataset")
    parser.add_argument("-t", "--target", type=int, default=-1, help="give a targeted class to attack our model")
    args = parser.parse_args()

    if args.dataset == 1:
        print("Loading cifar10 to the memory...")
        train_dset = datasets.CIFAR10(root = './datasets', train = True, download = False)
        test_dset = datasets.CIFAR10(root = './datasets', train = False, download = False)
        num_channels = 3
        num_paddings = 1
        my_morm = Normalize()
    elif args.dataset == 0:
        print("Loading mnist to the memory...")
        train_dset = datasets.MNIST(root='./datasets', train=True, download=False)
        test_dset = datasets.MNIST(root='./datasets', train=False, download=False)
        num_channels = 1
        num_paddings = 3

    if args.model == 1:
        print("Initialize ResNet18...")
        my_model = base_model.myResNet18(pretrained=False, input_channels=num_channels, first_padding=num_paddings, num_classes=10)
    elif args.model == 0:
        print("Initialize 2 Layer CNN...")
        my_model = base_model.two_layer_CNN(input_channels=num_channels, first_padding=num_paddings, num_classes=10)

    if isinstance(test_dset.data, np.ndarray):
        any_picture = test_dset.data[args.sample]
    else:
        any_picture = test_dset.data[args.sample].numpy()
    any_picture_target = test_dset.targets[args.sample]
    PIL_any_picture = Image.fromarray(any_picture.astype(np.uint8))
    transform_image = test_transform()(PIL_any_picture)[None, :, :, :]
    print("The chosen picture's idx and class are as below:")
    print(any_picture_target)
    print(test_dset.classes[any_picture_target])
    print("Visualize this picture:")
    visualize(any_picture)


    my_datasets = {"train": My_Dataset(train_dset, transform=train_transform(image_size=train_dset.data.shape[1])),
                   "test": My_Dataset(test_dset, transform=test_transform())}
    dset_loaders = {"train": data.DataLoader(dataset=my_datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=2),
                    "test": data.DataLoader(dataset=my_datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=2)}

    if args.norm == 0:
        my_morm = Normalize()
    elif args.norm == 1:
        datasets_mean, datasets_std = my_datasets["train"].compute()
        my_morm = Normalize(mean=datasets_mean, std=datasets_std)
    experiment_op = Experiment_Operator(datasets_loaders=dset_loaders,
                                        norm=my_morm,
                                        model=my_model,
                                        batch_size=args.batch_size,
                                        milestones=None,
                                        lr=args.lr,
                                        scale=1.0,
                                        is_BN=True,
                                        is_gpu=True)
    print("Loading learned parameters.")
    # "model_weights/cifar10-imagenet-resnet18.pth"
    if args.dataset == 0 and args.model == 0:
        model_weights_path = "model_weights/mnist-2layer-cnn.pth"
    elif args.dataset == 0 and args.model == 1:
        model_weights_path = "model_weights/mnist-resnet18.pth"
    elif args.dataset == 1 and args.model == 0:
        model_weights_path = "model_weights/cifar10-2layer-cnn.pth"
    elif args.dataset == 1 and args.model == 1:
        model_weights_path = "model_weights/cifar10-imagenet-resnet18.pth"
    experiment_op.load_model(path = model_weights_path)
    training_loss, testing_loss, training_acc, testing_acc = experiment_op.test()
    print("Print model's performance: training_accuracy = %.4f, testing_accuracy = %.4f" % (training_acc, testing_acc))
    print("                           training_loss = %.6f, testing_loss = %.6f" % (training_loss, testing_loss))


    pred = experiment_op.model(experiment_op.norm(transform_image.cuda()))
    #
    minus_ln_p = experiment_op.criterion(pred, torch.LongTensor([any_picture_target]).cuda())
    print("Prediction before attack:")
    print("True class probability:", np.exp(-minus_ln_p.item()))
    print("Predictive class:", pred.max(dim = 1)[1].item())

    print("Start attacking...")
    delta = experiment_op.sample_attack_train(transform_image.cuda(),
                                              torch.LongTensor([any_picture_target]).cuda(),
                                              epsilon=2.0 / 255,
                                              iterations=50,
                                              targeted_attack=torch.LongTensor([args.target]).cuda())

    print("Visualization of delta:")
    visualize(np.squeeze((50 * delta.detach().cpu().numpy() + 0.5)[0].transpose(1, 2, 0)))
    print("Visualization of sample+delta:")
    visualize(np.squeeze((transform_image + delta.detach().cpu())[0].numpy().transpose(1, 2, 0)))

