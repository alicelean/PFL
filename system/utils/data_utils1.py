import numpy as np
import os
import torch
import sys
from utils.vars import Programpath

def getlabelvector(label):
    '''
    将label变成一个10维向量，在对应index中的数字为标签index的样本个数
    '''
    v=[]
    for i in range(10):
        v.append(0)
    for i in label:
        v[i]+=1
    return v

import random

def generate_Randomnums(k, total_sum,batch_size=10):
    '''
    生成k个随机正整数，且这些数的和=total_sum
    同时保证每个client不小于k个数，至少有一个batch_size 的数据
    '''
    #print(" ------------------------------------------------------------------------------------------data  generate_Randomnums -----------------------------")
    numlist = []
    length=k
    origin_sum=total_sum
    while total_sum > 0 and length>1:
        imax= int(total_sum / length)
        #10 是batch_size
        numlist.append(random.randint(batch_size, imax))
        total_sum -= numlist[k-length]
        length-=1
    #print(k,len(result))
    numlist.append(origin_sum - sum(numlist))
    #print(k, len(result))
    if sum(numlist) != origin_sum:
        print("generate_numbers ERROR",sum(numlist) ,origin_sum)
    #print("generate_Randomnums is ",len(numlist),"个 client, each num list is:")
    #print(numlist)
    return numlist


def select_k_unique_elements(A, k, B):
    '''
    从A中每次选取k个数，并且不选已经在B中的数，B中存放已经选过的数据
    '''
    #print(" ------------------------------data  select_k_unique_elements -----------------------------")

    selected = set()
    while len(selected) < k:
        element = random.choice(A)
        if element not in B and element not in selected:
            selected.add(element)
    return list(selected)

def RandomShuffledata(alldata,client_nums):
    #print(" ------------------------------------------------------------------------------------------data  RandomShuffledata -----------------------------")

    '''
    #将一个list alldata 分割成client_nums个子list
    '''
    if len(alldata)<=client_nums:
        print("Shuffledata client_nums ERROR","data length is ",len(alldata),"client num is ",client_nums)
    # Shuffle A randomly
    random.shuffle(alldata)
    # 随机生成一个向量
    each_client_datanums_list=generate_Randomnums(client_nums, len(alldata))

    allindex =list(range(len(alldata)))
    selectedindex = []
    shuffledata=[]
    i=0
    for num in each_client_datanums_list:
        client_data = []
        client_index= select_k_unique_elements(allindex, num, selectedindex)
        i+=1
        for index in client_index:
            client_data.append(alldata[index])
            #已经选过的index放回数据中，下次则不再选
            selectedindex.append(index)
        #print("client ", i + 1, "data:", num, "alldata is:", len(alldata), len(client_data),type(client_data[0]))
        shuffledata.append(client_data)
    #print("shuffle",shuffledata[0])

    #判定result数据
    sums=0
    for clienti in shuffledata:
        sums+=len(clienti)
    if sums !=len(alldata) or len(shuffledata)!=len(each_client_datanums_list):
       if len(shuffledata) == len(each_client_datanums_list)+1:
           print("ssssss",len(shuffledata),len(each_client_datanums_list))

       else:
            print("Shuffledata  SHUFFLE ERROR: ","shuffledata length is ",len(shuffledata),"but client_nums is ",len(each_client_datanums_list),client_nums)
    else:
        print("shuffle data ",len(shuffledata),"个 list ,total length is ",len(alldata))
        #print("result[0][0] example is ",result[0][0])

    return shuffledata,each_client_datanums_list
def change_data (datalist):
    '''
    转换数据类型
    input:[(x,y)]
    output: data={'x':numpy.ndarry(874,1,28,28),'y':numpy.ndarry(6)}
    '''

    x_array=[]
    y_array=np.array([])
    #print(type(datalist))
    for sample in datalist:
        #print(sample[0].shape)
        x=sample[0].tolist()
        x_array.append(x)
        y=sample[1].numpy().astype(np.int64)
        y_array = np.append(y_array, y)

    # print("sample type", type(x[0]), x[0],y_array.shape)
    x_array1=np.array(x_array)
    #print("change data type",x_array1.shape,y_array.shape)
    return {'x': x_array1, 'y': y_array}

def read_data_new(dataset,num_clients,filedir,is_train):
    '''
    重新分配节点的数据(self.dataset, self.num_clients, is_train=True)
    '''
    #print(" ------------------------------------------------------------------------------------------data  read_data_new -----------------------------")

    alldata=[]
    label=[]
    for i in range(20):
        # data={'x':numpy.ndarray,,'y':numpy.ndarray}
        # data['x'].shape is(1972, 1, 28, 28),client_data_num=1972
        # data['y'].shape is (1972, )
        data=read_data(dataset, i,filedir, is_train)
        #print("read_data origin data", data['x'].shape,data['y'].shape)
        #变换成张量形式
        X= torch.Tensor(data['x']).type(torch.float32)
        Y= torch.Tensor(data['y']).type(torch.int64)
        # 整体汇集成一个元组list，特征和标签数据相对应
        data = [(x, y) for x, y in zip(X, Y)]
        labeldata=[y.tolist() for x, y in zip(X, Y)]
        #所有数据拷贝到alldata
        for k in range(len(data)):
            alldata.append(data[k])
            label.append(labeldata[k])

    #print("alldata is ",len(alldata),"example is ",alldata[0])
     #将数据随机分割成num_clients个客户端数据
    shuffledata,each_client_datanums_list=RandomShuffledata(alldata,num_clients)

    #print(f"shuffledata get success,distrubution is {each_client_datanums_list}")

    return shuffledata



# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset, idx,filedir, is_train=True):
    if is_train:

        train_data_dir = os.path.join(Programpath+'/dataset', dataset, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:

        test_data_dir = os.path.join(Programpath+'/dataset', dataset, 'test/')
        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data_new(dataset, idx,filedir, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx,filedir, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        #增加代码段-----------------
        trainlabel = [y.tolist() for x, y in zip(X_train, y_train)]
        v = getlabelvector(trainlabel)
        # 增加代码段-----------------
        return train_data,v
    else:
        test_data = read_data(dataset, idx,filedir, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        # 增加代码段-----------------
        testlabel = [y.tolist() for x, y in zip(X_test, y_test)]
        v = getlabelvector(testlabel)
        # 增加代码段-----------------
        return test_data,v


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data




