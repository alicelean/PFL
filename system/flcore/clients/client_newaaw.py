import torch,copy
import torch.nn as nn
import numpy as np
import time,statistics
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientNewAAW(Client):
    def __init__(self, args, id, traindata, testsdata, train_samples, test_samples, **kwargs):
        super().__init__(args, id, traindata, testsdata, train_samples, test_samples, **kwargs)
        #没有更新的梯度累计和
        self.prev_grad=None
        self.AAweight=None
        self.hasinit=False
        self.initweight=0
        #前一次的模型参数
        self.prev_params=None
        #所有的梯度*参数之和
        self.grad_para_sum=None
        self.threshold=0.01
        #是否更新权重
        self.update =True
        self.eta=0.01


    def train(self):
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
        params = list(self.model.parameters())
        self.currentgrad = [(torch.ones_like(param.data) * 0).to(self.device) for param in params]
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
                #保留梯度-----------------

                for para_g, grad in zip(params, self.currentgrad):
                    grad.data = grad + torch.mul(para_g.grad, 0.001)
                #print(self.currentgrad)
                # 保留梯度-----------------
                # 根据梯度更新模型的参数
                self.optimizer.step()

        # self.model.cpu()
        # 如果启用了学习率衰减（self.learning_rate_decay为True），则调用self.learning_rate_scheduler.step()进行学习率的更新

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        #保留上一次的模型参数
        model_t = copy.deepcopy(self.model)
        self.prev_parames = list(model_t.parameters())
        #self.update_prev_params()

        # 如果启用了隐私保护（self.privacy为True），则打印当前客户端的隐私参数信息，包括隐私预算 epsilon 和 sigma
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def get_prev_grads(self):
        self.prev_grad = [(torch.ones_like(param.data) * 0).to(self.device) for param in self.model.parameters()]
        #print("self.prev_grad",self.prev_grad)
        current_learning_rate = self.eta
        rand_loader = DataLoader(self.traindata, len(self.traindata), drop_last=True)
        params= list(self.model.parameters())  # 获取全局模型的参数列表
        #self.prev_grad = [(torch.ones_like(param.data) * 0).to(self.device) for param in params]
        optimizer = torch.optim.SGD(params, lr=0)  # 使用全局模型的参数列表创建优化器
        start_time = time.time()
        aloss = []
        for x, y in rand_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            output = self.model(x)  # 使用全局模型计算输出
            loss_value = self.loss(output, y)  # 计算损失值
            loss_value.backward()  # 反向传播计算梯度
            # 此时可以获取梯度信息
            aloss.append(loss_value.item())
            for para_g, grad in zip(params,self.prev_grad):
                grad.data = grad + torch.mul(para_g.grad, current_learning_rate)
            # # 计算梯度张量的平均范数
            # mean_gradient_norm = sum(torch.norm(g) for g in self.updategrad) / len(self.updategrad)
            # # 判断平均范数是否小于阈值
            # if mean_gradient_norm < 0.00001:
            #     print("平均梯度范数小于阈值，停止训练")
            #     self.stop = True

        update_time = time.time() - start_time
        #说明聚合后的模型方向是错误的，此时应该使用前一次的聚合权重即可，也就是聚合权重不更新
        if len(self.losslist)>0:
            if statistics.mean(aloss)>  self.losslist[-1] or np.std(self.losslist) < self.threshold:
                self.update=False



        self.losslist.append(statistics.mean(aloss))
        print('Client:', self.id, 'current mean(aloss):', statistics.mean(aloss),'pre mean loss',self.losslist[-1],  '\tStd:', np.std(aloss),
                  'self.threshold:',self.threshold, '\tAAW epochs:')
        self.timecost.append(update_time)

    def updateweight(self):
        '''
        根据梯度更新权重
        Returns:

        '''
        #print(f"client {self.id} before weight ",self.AAweight)
        # for weight, grad in zip(self.AAweight, self.currentgrad):
        #     weight.data = weight - torch.mul(grad, current_learning_rate)

        for grad, param, weight in zip(self.prev_grad,
                                                   self.prev_params, self.AAweight):
           # print("grad",grad)
            #print("param",param)
            weight.data = weight -grad*param

        if self.id==1:
            print(f"client {self.id} after update weight is", self.AAweight)

    def init_weight(self, global_model: nn.Module):
        '''
        利用js距离初始化聚合参数，形状与globalmodel相似
        Args:
            global_model:

        Returns:

        '''
        if self.initweight==0:
            raise ValueError("ERROR:AAW.self.initweight is zero")
        params = list(global_model.parameters())
        #print(f"client  {self.id} self.initweight   {self.initweight}")
        self.AAweight = [torch.mul(torch.ones_like(param.data), self.initweight).to(self.device) for param in params]
        #print("INit AAweight is  -----------------------",self.AAweight)




    def init_prev_params(self, model):
        '''
        将model 复制给本地模型
        Args:
            model:

        Returns:

        '''
        if self.prev_params == None:
            self.prev_params = [torch.ones_like(param.data).to(self.device) for param in model.parameters()]
            self.prev_grad = [(torch.ones_like(param.data) * 0).to(self.device) for param in model.parameters()]
        #复制模型参数
        for new_param, old_param in zip(model.parameters(), self.prev_params):
            old_param.data = new_param.data.clone()

    def update_prev_params(self):
        '''
        将model 复制给本地模型
        Args:
            model:

        Returns:

        '''
        # 复制更新后的模型参数
        for new_param, old_param in zip(self.model.parameters(), self.prev_params):
            old_param.data = new_param.data.clone()













