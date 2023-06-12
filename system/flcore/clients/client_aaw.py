import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA
from utils.AAW import AAW


class clientAAW(Client):
    def __init__(self, args, id, traindata,testsdata,train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata,testsdata, train_samples, test_samples, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        #新增5个属性，
        self.traindata=traindata
        self.testsdata=testsdata

        self.distance=0
        self.alldistance=0
        self.alllabel=None
        # self.hasinit = False
        self.adptive_elta=None
        # aaw新增属性

        self.average_grad =None
        self.allgrad =None
        self.AAW = AAW(self.sizerate, args.num_clients, self.id, self.loss, self.traindata, self.batch_size,
                       self.rand_percent, self.layer_idx, self.eta, self.device)

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size,
                       self.rand_percent, self.layer_idx, self.eta, self.device)

    def train(self):
        #print(f"***************************client {self.id}     client_aaw train***************************")
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        #增加记录梯度的代码---------
        params_l= list( self.model.parameters())
        self.initgrad(self.AAW.global_model)
        if self.allgrad is None:
            self.allgrad=[(torch.ones_like(param.data) * 0).to(self.device) for param in params_l]

        #-----------------------
        start_time = time.time()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                # 记录所有的梯度，#增加记录梯度的代码---------
                for para, avgrad in zip(params_l, self.average_grad):
                    avgrad.data = avgrad + para.grad
                # ------------------------------
                self.optimizer.step()


        # for para, allg in zip(params_l, self.allgrad):
        #     # 利用全局梯度对模型参数的更新方向进行控制
        #     para.data = para - 0.0001 * allg





        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        #print(f"***************************client {self.id}     end training***************************")

    def local_initialization(self, received_global_model,round):
        #ALA模块
        #self.ALA.adaptive_local_aggregation(received_global_model, self.model)
        # 非ALA模块，一种静态的权重来生成本地的初始化模型
        #self.ALA.adaptive_local_aggregation_aaw(received_global_model, self.model)
        # 分层存储
        #self.ALA.static_local_aggregation(received_global_model, self.model)
        self.AAW.adaptive_aggregation_weight(received_global_model, self.model,round)


    def setlabel(self):
        #train_data = [(x, y) for x, y in zip(X_train, y_train)]
        label=[]
        for data in self.traindata:
            label.append(data[1].tolist())
        for data in self.testsdata:
            label.append(data[1].tolist())
        for i in label:
            self.label[i] += 1
        print(f"client alajs  client {self.id} label is {self.label}")
    def initgrad(self,globalmodel):
        '''
        记录每次训练获得的梯度和，用于修正其他client的本地模型更新方向
        Args:
            globalmodel:

        Returns:

        '''
        params_g = list(globalmodel.parameters())
        self.average_grad=[(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]







