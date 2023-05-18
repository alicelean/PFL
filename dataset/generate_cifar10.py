import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "Cifar10/"


# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition):
    '''
    生成 CIFAR-10 数据集，并将数据按照指定的方式进行划分和保存。该函数的输入参数包括：
        Args:
            dir_path: 数据集保存的目录路径
            num_clients: 用户的数量
            num_classes: 数据集中的类别数量
            niid: 一个布尔值，表示是否生成非独立同分布（Non-IID）的数据集
            balance: 一个布尔值，表示是否平衡分配数据给用户
            partition: 数据集划分的方式，可以是 "homo"（同质划分）或者 "hetero"（异质划分）
        Returns:
    '''


    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    #check 函数的作用是检查数据集和配置文件是否已经存在。如果数据集和配置文件已经存在，即已经生成过，则返回 True，并且不再执行后续的数据集生成和划分操作。
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar10 data,
    # 定义了一个数据预处理的转换（transform）操作，将原始的 CIFAR-10 图像数据转换为神经网络可以处理的标准化的张量格式，transforms.ToTensor()：将图像数据转换为张量（Tensor）格式。这个操作将图像数据从 PIL 图像对象或 NumPy 数组转换为张量，并且将像素值缩放到 0 到 1 的范围内。transforms.Normalize(mean, std)：对图像数据进行归一化操作。这个操作将图像的每个通道进行标准化处理，使得图像在每个通道上的均值为 mean，标准差为 std。在这个例子中，均值和标准差都设置为 (0.5, 0.5, 0.5)，即将图像的每个通道的像素值都减去 0.5，并且除以 0.5，从而使得图像的像素值在 -1 到 1 的范围内。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #torchvision.datasets.CIFAR10 类来加载 CIFAR-10 数据集。这个类提供了访问 CIFAR-10 数据集的功能，并且可以根据需要进行下载和转换
    #dir_path + "rawdata"表示数据集根目录的路径,train：一个布尔值，表示是否加载训练集。当为 True 时加载训练集，当为 False 时加载测试集,download：一个布尔值，表示是否需要下载 CIFAR-10 数据集。当数据集文件不存在时，将自动下载,transform：数据集的转换操作，即之前定义的 transforms.Compose 对象，用于对图像数据进行预处理和转换
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    #使用 torch.utils.data.DataLoader 类来创建数据加载器，用于批量加载 CIFAR-10 数据集的训练集和测试集。DataLoader 类可以将数据集分割成批次（batches），并提供多线程加载数据的功能，以加快训练过程中数据的读取速度。
    #shuffle：一个布尔值，表示是否在每个 epoch（数据集遍历的一轮）之前对数据进行洗牌（随机打乱顺序）,在这里，由于已经对数据进行了划分，不再需要在加载时进行洗牌，因此设置为 False。
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    # enumerate 函数遍历 trainloader 和 testloader 中的批次数据,每次迭代中，enumerate 函数返回一个元组，包含一个索引和一个批次的数据,将批次的数据分别赋值给 trainset.data 和 trainset.targets，以及 testset.data 和 testset.targets。
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data
#将训练集和测试集的图像数据和标签分别存储在 dataset_image 和 dataset_label 中，并最终将它们转换为 NumPy 数组,extend 方法将训练集的图像数据和测试集的图像数据分别添加到 dataset_image 图像数据是张量格式，通过 cpu() 方法将其转移到 CPU 上，并通过 detach().numpy() 方法将其转换为 NumPy 数组
    dataset_image = []
    dataset_label = []
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition)