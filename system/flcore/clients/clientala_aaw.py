import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA
from utils.AAW import AAW


class clientALA_AAW(Client):
    def __init__(self, args, id, traindata,testsdata,train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata,testsdata, train_samples, test_samples, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        #新增5个属性，
        self.traindata=traindata
        self.testsdata=testsdata
        if args.dataset=='Cifar10':
            self.label = [0 for i in range(10)]
        elif  args.dataset=='Cifar100':
            self.label=[0 for i in range(100)]
        self.distance=0
        self.alldistance=0
        self.alllabel=None
        # self.hasinit = False
        self.adptive_elta=None
        # aaw新增属性
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
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        #print(f"***************************client {self.id}     end training***************************")

    def local_initialization(self, received_global_model,round):
        #print("------------clientala_aaw.local_initialization")
        #ALA模块
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)
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







