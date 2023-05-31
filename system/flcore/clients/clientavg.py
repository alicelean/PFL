import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, traindata,testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata,testsdata, train_samples, test_samples, **kwargs)

    def train(self):
        #加载训练数据集，返回一个数据加载器 trainloader。
        trainloader = self.load_train_data()
        # self.model.to(self.device)，将模型设置为训练模式
        self.model.train()

        # differential privacy，如果开启了隐私保护（self.privacy为True），则调用initialize_dp()函数对模型、优化器和数据加载器进行差分隐私初始化，返回更新后的模型、优化器、数据加载器和隐私引擎。
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_epochs
        #如果设置了self.train_slow，则在每轮训练中随机休眠一段时间。然后迭代训练数据加载器trainloader，对每个批次的输入数据 x 和标签 y 进行训练。
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
                #将优化器中的梯度置零
                self.optimizer.zero_grad()
                #进行梯度反向传播，计算模型参数的梯度
                loss.backward()
                #根据梯度更新模型的参数
                self.optimizer.step()

        # self.model.cpu()
        #如果启用了学习率衰减（self.learning_rate_decay为True），则调用self.learning_rate_scheduler.step()进行学习率的更新

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        #如果启用了隐私保护（self.privacy为True），则打印当前客户端的隐私参数信息，包括隐私预算 epsilon 和 sigma
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")