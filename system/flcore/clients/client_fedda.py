import copy

import torch
import torch.nn as nn
import numpy as np
import time,math
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientDA(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        self.lossvalue=None

    def train(self):
        #在上层，保留部分全局信息
        if self.prev_weight!=None:
            self.initnext_model()
        # 加载训练数据集，返回一个数据加载器 trainloader。
        trainloader = self.load_train_data()
        # self.model.to(self.device)，将模型设置为训练模式
        self.model.train()

        # differential privacy，如果开启了隐私保护（self.privacy为True），则调用initialize_dp()函数对模型、优化器和数据加载器进行差分隐私初始化，返回更新后的模型、优化器、数据加载器和隐私引擎。
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_steps = self.local_epochs
        # 如果设置了self.train_slow，则在每轮训练中随机休眠一段时间。然后迭代训练数据加载器trainloader，对每个批次的输入数据 x 和标签 y 进行训练。
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
                # 将优化器中的梯度置零
                self.optimizer.zero_grad()
                # 进行梯度反向传播，计算模型参数的梯度
                loss.backward()
                # 根据梯度更新模型的参数
                self.optimizer.step()

        # self.model.cpu()
        # 如果启用了学习率衰减（self.learning_rate_decay为True），则调用self.learning_rate_scheduler.step()进行学习率的更新

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        #记录当前模型
        model_t=copy.deepcopy(self.model)
        self.prev_weight = list(model_t.parameters())



        # 如果启用了隐私保护（self.privacy为True），则打印当前客户端的隐私参数信息，包括隐私预算 epsilon 和 sigma
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

        #评估阶段
        cl, ns = self.train_metrics()
        #print(f"client {self.id},current round is {self.current_round},loss:{cl},{ns}")
        #losses.append(cl * 1.0)
        self.lossvalue=math.log(cl*1.0)*math.log(cl*1.0)


    def initnext_model(self):
        params_g = list(self.model.parameters())
        #print("before ",params_g[-self.layer:])
        # #保留底层全局模型的数据
        # for param, param_g in zip(self.prev_weight[:-self.layer], params_g[:-self.layer]):
        #     param.data = param_g.data.clone()
        # 更新上层数据为本地模型与全局模型的差异，这样的话，上层模型是被修正的本地模型，下层是全局模型，整体既保留了局部信息，又保留了全局信息。
        for param, param_g in zip(self.prev_weight[-self.layer:], params_g[-self.layer:]):
            param_g.data = param+(param_g - param) * 0.5
        #params_t= list(self.model.parameters())
        #print("after ", params_t[-self.layer:])