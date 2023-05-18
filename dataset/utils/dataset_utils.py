import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = batch_size / (1-train_size) # least samples for each client
alpha = 0.1 # for Dirichlet distribution

def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, 
        balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    '''
    将数据集按照指定的方式划分为多个客户端，并进行数据的非独立和平衡性处理
    Args:
        data:一个元组，包含图像数据和标签数据。
        num_clients:客户端数量
        num_classes:数据集中的类别数量
        niid:一个布尔值，表示是否进行非独立分布的划分
        balance:一个布尔值，表示是否进行数据的平衡划分
        partition:数据集划分的方式
        class_per_client:每个客户端的类别数量
    Returns:
        X：一个列表，包含划分后的图像数据。其中每个元素对应一个客户端的图像数据
        y：一个列表，包含划分后的标签数据。其中每个元素对应一个客户端的标签数据。
        statistic：一个列表，其中每个元素对应一个客户端的统计信息。

    '''
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    #解包操作，将元组中的内容分别赋值给 dataset_content 和 dataset_label
    dataset_content, dataset_label = data
    #存储每个客户端对应的数据索引
    dataidx_map = {}
   #如果 niid 为 False，表示进行非独立分布的划分，即每个客户端可以包含来自不同类别的样本。在这种情况下，将 partition 设置为 'pat'，表示按照类别进行划分。将 class_per_client 设置为 num_classes，表示每个客户端包含所有的类别,在非独立分布的情况下，确保每个客户端都包含完整的类别信息
    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        #创建了一个从 0 到 len(dataset_label)-1 的数组
        idxs = np.array(range(len(dataset_label)))
        #idx_for_each_class 列表的长度为类别数量 num_classes，每个元素是一个数组，包含了对应类别的样本索引。
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])


        #确定每个类别应分配到哪些客户端，并确保每个客户端包含的类别数量不超过指定的值--------
        #class_num_per_client每个客户端应包含的类别数量为class_per_client
        class_num_per_client = [class_per_client for _ in range(num_clients)]
        #迭代每个类别，选择将该类别分配给哪些客户端，遍历每个客户端，检查该客户端可以包含的类别数量是否大于零。如果是，则将该客户端的索引添加到 selected_clients 列表中
        #通过切片操作，从 selected_clients 列表中选择前面一部分客户端，使其包含的类别数量接近 ，num_clients/num_classes*class_per_client。这样可以实现尽可能平均地将类别分配给客户端。
        #通过这段代码，可以动态地将类别分配给不同的客户端，以实现数据的非独立分布。
        #num_clients/num_classes*class_per_client：一个类别属于多少个客户端，将类别顺序分给客户端
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/num_classes*class_per_client)]

           #idx_for_each_class[i]---类别i对应的样本索引list,num_all_samples类别i的样本数量。num_selected_clients：类别i分配给客户端的数量
            #num_per：每个客户端可以获得的类别i的样本个数
            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
             #计算每个客户端应分配的样本数量列表 num_samples。
            if balance:
                #确保每个客户端分配的样本数量接近平均值 num_per，同时保持总样本数量的一致性，先生成num_selected_clients-1个客户端对应的样本数量
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                #使用 np.random.randint 函数生成 num_selected_clients-1 个随机数，来分配剩余的样本数量。
                #max(num_per/10, least_samples/num_classes)--最小值保证客户端的最少数据量，num_per/10：将每个客户端应分配的样本数量 num_per 除以 10。这样做的目的是将样本数量的最小值设置为每个客户端应分配样本数量的十分之一。通过这样的设置，可以确保每个客户端至少获得总样本数量的一小部分，least_samples/num_classes：将 least_samples（样本数量的最小要求）除以 num_classes（类别数量）。这个计算的结果表示每个类别应至少分配给客户端的样本数量。通过这样的设置，可以确保每个类别的样本数量在分配过程中得到平衡
                #选择较大的值作为每个客户端应分配的样本数量的最小值。这样做的目的是确保每个客户端获得的样本数量不会过少，并尽量保持样本分配的平衡性。
                num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            #将划分好的样本索引分配给每个客户端，dataidx_map：client:[]
            idx = 0
            #对于client 要分配num_sample个样本
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                #已经给client分配了一个类别数据，client还需要的类别数量-1
                class_num_per_client[client] -= 1

    elif partition == "dir":
        #独立分布的划分，即每个客户端只包含特定类别的样本。在这种情况下，将 partition 设置为其他具体的划分方式，例如 'dir'，并根据具体需求设置 class_per_client 的值。
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            #idx_batch，其中包含 num_clients 个空列表，用于存储每个客户端的样本索引
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                #返回一个数组，包含所有标签为2的样本的索引。 np.where(dataset_label == k)[0]表示在 dataset_label 中找到所有等于 k 的元素，并返回它们的索引。
                idx_k = np.where(dataset_label == k)[0]
                #随机打乱list
                np.random.shuffle(idx_k)

                #dirichlet 函数用于生成符合狄利克雷分布的随机数。一种多维概率分布，常用于生成多类别的随机变量。
                #给定参数 alpha 和样本数目 num_clients ,num_clients可以表示维度，np.repeat(alpha, num_clients)：将 alpha 中的每个元素重复 num_clients 次而得到。
                # 生成一个长度为 num_clients 的数组,每个元素表示一个随机样本，符合狄利克雷分布,alpha 是一个正数数组，用于指定每个维度的参数值，np.repeat(alpha, num_clients) 将 alpha 数组进行复制，使其长度与 num_clients 相同
                #np.random.dirichlet()它的参数是一个正数数组 alpha，表示每个维度的权重或浓度，np.random.dirichlet(alpha) 将返回一个随机样本，符合狄利克雷分布，其中样本的维度与 alpha 数组的长度相同
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                print("proportions",proportions)

                #idx_j:client j 的样本索引列表，将数据集样本数量 N 平均分配给 num_clients 个客户端，根据当前客户端已经包含的样本数量，动态调整该客户端所占比例 p 的值，当客户端已经包含的样本数量小于每个客户端应分配的样本数时，该客户端的比例保持不变 (p)；否则，该客户端的比例被设置为0，即不再接收更多的样本。确保每个客户端接收到的样本数量不超过其应分配的样本数。
                #在数据集划分过程中，可能会出现一些客户端已经接收到足够数量的样本，而其他客户端仍需要更多样本的情况。通过这个条件表达式，可以在采样时限制每个客户端接收样本的数量，以实现平衡性的划分。
                proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                #表中的元素进行归一化处理，将比例值调整为相对比例，以便在划分样本时能够按比例分配给每个客户端
                proportions = proportions/proportions.sum()
                #累积求和操作通常用于计算累积概率分布、累积和等方面，将累积和按比例进行缩放。将浮点数转换为整数，去掉最后一个元素
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                print(f"proportions len:{len(proportions)},num_clients:{num_clients}")
                #np.split(idx_k,proportions)：根据 proportions 的值，将 idx_k 分割成了多个子数组，每个子数组由 proportions 中的相应位置指定
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                #用于记录 idx_batch 中所有子数组的最小长度。
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data,根据index提取样本和标签数据
    for client in range(num_clients):
        #客户端对应的索引列表 idxs,根据客户端的索引列表 idxs，从原始数据集 dataset_content 中提取相应的样本，并将提取的样本赋值给 X[client]
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        #计算每个客户端中每个类别的样本数量，并将统计结果添加到 statistic[client] 列表中，np.unique(y[client])获取客户端 client 的目标标签 y 中的唯一类别
        #将类别和对应的样本数量作为元组 (int(i), int(sum(y[client]==i)))
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))
            

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 100)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")



