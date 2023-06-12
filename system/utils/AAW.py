import numpy as np
import torch
import torch.nn as nn
import copy,statistics
import random,time
from torch.utils.data import DataLoader
from typing import List, Tuple
import math
import tensorflow as tf
class AAW:
    def __init__(self,
                sizerate:float,
                num_clients: int,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple],
                batch_size: int,
                rand_percent: int,
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu',
                threshold: float = 0.01,
                num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID.
            loss: The loss function.
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.01
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.initweight = 0
        self.sizerate=sizerate
        self.num_clients = num_clients
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.AAweights = None # Learnable local aggregation weights.
        self.timecost = []
        self.notadptive=False
        self.start_phase = True
        self.updategrad=[]
        self.hasinit=False
        self.global_model=None
        self.stop=False
        self.etaj=0.01
        self.learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    self.etaj, 100)


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
        #print(f"client  {self.cid} self.initweight   {self.initweight}")
        self.AAweights = [torch.mul(torch.ones_like(param.data), self.initweight).to(self.device) for param in params]
       # print("INit AAweight is  -----------------------", type(self.AAweights))








    def adaptive_aggregation_weight(self, global_model: nn.Module, local_model: nn.Module,round :int) -> None:
        """
        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.
        """
        if self.notadptive or self.stop:
            return

        if not self.hasinit:
            self.init_weight(self.global_model)
            self.hasinit = True

        rand_ratio = self.rand_percent / 100  # 随机采样比例
        rand_num = int(rand_ratio * len(self.train_data))  # 计算随机采样的样本数量
        if rand_num < self.batch_size:
            print(
                f"ERROR: example rand_num={rand_num} < self.batch_size={self.batch_size} because len(self.train_data)={len(self.train_data)}, rand_ratio={rand_ratio}")
            return
        rand_idx = random.randint(0, len(self.train_data) - rand_num)  # 随机选择采样的起始索引

        #使用所有数据集
        #rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=True)
        rand_loader = DataLoader(self.train_data, len(self.train_data), drop_last=True)

        params_g = list(global_model.parameters())  # 获取全局模型的参数列表
        params_l = list(local_model.parameters())  # 获取本地模型的参数列表
        local_model_t = copy.deepcopy(global_model)
        params_l_t = list(local_model_t.parameters())

        global_model_t = copy.deepcopy(global_model)
        params_g_t = list(global_model_t.parameters())
        optimizer = torch.optim.SGD(params_g_t, lr=0)  # 使用全局模型的参数列表创建优化器

        self.updategrad=[(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]

           # print("w", self.AAweights)
            # 如果是第一次更新，初始化 AAweight 为元素值为 init_value 的张量

           # 使用随机采样的数据创建 DataLoader

        losses = []  # 记录损失值
        cnt = 0  # 权重训练迭代计数器
        samples = 0
        #print(f"client {self.cid} start  aaweight learning------------------------")


        # 记录更新前的时间戳
        start_time = time.time()
        #prev_AAweights = [aaweight.clone() for aaweight in self.AAweights]  # 复制之前的self.AAweights
        #print("before adptive", self.AAweights)
        aloss=[]
        for x, y in rand_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            output = global_model_t(x)  # 使用全局模型计算输出
            loss_value = self.loss(output, y)  # 计算损失值
            loss_value.backward()  # 反向传播计算梯度
            #此时可以获取梯度信息
            #print(f"y,{y},loss is:",loss_value.item())
            aloss.append(loss_value.item())
            #losses.append(loss_value.item())
            etaj=0.01
            for para_g, para_l_t, aaweight,upgrad in zip(params_g_t, params_l_t, self.AAweights,self.updategrad):
                #aaweight.data = aaweight - torch.mul(para_g.grad, 0.01 * self.sizerate) * para_l_t
                aaweight.data = aaweight - torch.mul(para_g.grad,   self.etaj* self.sizerate)
                upgrad.data = upgrad + torch.mul(para_g.grad,   self.etaj* self.sizerate)
                # print("self.sizerate",self.sizerate)
                # print("para_g.grad",para_g.grad)
                # print("para_l_t",para_l_t)
                #print("torch.mul(para_g.grad, 0.01 * self.sizerate) * para_l_t", torch.mul(para_g.grad, 0.01 * self.sizerate) * para_l_t)

                    #weight.data=weight-torch.mul(para_g.grad, self.eta * self.datasizerate) * para_l_t



                    # print("self.eta * self.datasizerate", self.eta * self.sizerate )
                    # print("self.eta ", self.eta )
                    # print(" self.datasizerate", self.sizerate)
                    # print("para_g.grad", para_g.grad)
                    # print("para_l_t", para_l_t)
                    # if torch.equal(aaweight, updated_aaweight):
                    #     print("aaweight和更新后的AAweights的元素相同")
                    # else:
                    #     print("aaweight和更新后的AAweights的元素不同")
                    # self.AAweights[i] = updated_aaweight
                    # i=i+1


                # 更新 AAweight，根据全局模型在本地数据上的梯度和本地模型的参数

        # 计算梯度张量的平均范数
        mean_gradient_norm = sum(torch.norm(g) for g in self.updategrad) / len(self.updategrad)
        # 判断平均范数是否小于阈值
        if mean_gradient_norm < 0.00001:
            print("平均梯度范数小于阈值，停止训练")
            self.stop=True
        if len(aloss)==0:
            print("aloss {aloss},rand_loader :{len(rand_loader)}")
        losses.append(statistics.mean(aloss))
        update_time = time.time() - start_time
        #print('Client:', self.cid,'mean(aloss):',statistics.mean(aloss), '\tStd:', np.std(aloss),'self.threshold:',self.threshold,'\tAAW epochs:', cnt)
        #print("after", self.AAweights[0])

        # for tensor1, tensor2 in zip(prev_AAweights, self.AAweights):
        #     if torch.allclose(tensor1, tensor2):
        #         print("newweight和初始的AAweights的元素相同")
        #     else:
        #         print("newweight和初始的AAweights的元素不同")

        # 计算更新所使用的时间
        update_time = time.time() - start_time
        self.timecost.append(update_time)
        #print("更新所使用的时间:", update_time, "秒")
        #print(f"client {self.cid} end   learning-----------------------")

        #print("Updated AAweight", type(self.AAweights))

        if statistics.mean(aloss)<0.5:
            self.stop = True

    def adaptive_aggregation_weight_update(self, global_model: nn.Module, local_model: nn.Module, round: int) -> None:
        """
        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.
        """


        if self.notadptive or self.stop:
            return

        current_learning_rate=0.001


        if not self.hasinit:
            self.init_weight(self.global_model)
            self.hasinit = True

        rand_ratio = self.rand_percent / 100  # 随机采样比例
        rand_num = int(rand_ratio * len(self.train_data))  # 计算随机采样的样本数量
        if rand_num < self.batch_size:
            print(
                f"ERROR: example rand_num={rand_num} < self.batch_size={self.batch_size} because len(self.train_data)={len(self.train_data)}, rand_ratio={rand_ratio}")
            return
        rand_idx = random.randint(0, len(self.train_data) - rand_num)  # 随机选择采样的起始索引

        # 使用所有数据集
        # rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=True)
        rand_loader = DataLoader(self.train_data, len(self.train_data), drop_last=True)

        params_g = list(global_model.parameters())  # 获取全局模型的参数列表
        params_l = list(local_model.parameters())  # 获取本地模型的参数列表
        local_model_t = copy.deepcopy(global_model)
        params_l_t = list(local_model_t.parameters())

        global_model_t = copy.deepcopy(global_model)
        params_g_t = list(global_model_t.parameters())
        optimizer = torch.optim.SGD(params_g_t, lr=0)  # 使用全局模型的参数列表创建优化器

        self.updategrad = [(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]

        # print("w", self.AAweights)
        # 如果是第一次更新，初始化 AAweight 为元素值为 init_value 的张量

        # 使用随机采样的数据创建 DataLoader

        losses = []  # 记录损失值
        cnt = 0  # 权重训练迭代计数器
        samples = 0
        # print(f"client {self.cid} start  aaweight learning------------------------")

        # 记录更新前的时间戳
        start_time = time.time()
        # prev_AAweights = [aaweight.clone() for aaweight in self.AAweights]  # 复制之前的self.AAweights
        # print("before adptive", self.AAweights)
        aloss = []
        for x, y in rand_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            output = global_model_t(x)  # 使用全局模型计算输出
            loss_value = self.loss(output, y)  # 计算损失值
            loss_value.backward()  # 反向传播计算梯度
            # 此时可以获取梯度信息
            # print(f"y,{y},loss is:",loss_value.item())
            aloss.append(loss_value.item())
            # losses.append(loss_value.item())

            for para_g, para_l_t, aaweight, upgrad in zip(params_g_t, params_l_t, self.AAweights, self.updategrad):
                # aaweight.data = aaweight - torch.mul(para_g.grad, 0.01 * self.sizerate) * para_l_t
                aaweight.data = aaweight - torch.mul(para_g.grad, current_learning_rate * self.sizerate)
                upgrad.data = upgrad + torch.mul(para_g.grad, current_learning_rate * self.sizerate)
                # print("self.sizerate",self.sizerate)
                # print("para_g.grad",para_g.grad)
                # print("para_l_t",para_l_t)
                # print("torch.mul(para_g.grad, 0.01 * self.sizerate) * para_l_t", torch.mul(para_g.grad, 0.01 * self.sizerate) * para_l_t)

                # weight.data=weight-torch.mul(para_g.grad, self.eta * self.datasizerate) * para_l_t

                # print("self.eta * self.datasizerate", self.eta * self.sizerate )
                # print("self.eta ", self.eta )
                # print(" self.datasizerate", self.sizerate)
                # print("para_g.grad", para_g.grad)
                # print("para_l_t", para_l_t)
                # if torch.equal(aaweight, updated_aaweight):
                #     print("aaweight和更新后的AAweights的元素相同")
                # else:
                #     print("aaweight和更新后的AAweights的元素不同")
                # self.AAweights[i] = updated_aaweight
                # i=i+1

                # 更新 AAweight，根据全局模型在本地数据上的梯度和本地模型的参数

        # 计算梯度张量的平均范数
        mean_gradient_norm = sum(torch.norm(g) for g in self.updategrad) / len(self.updategrad)
        # 判断平均范数是否小于阈值
        if mean_gradient_norm < 0.00001:
            print("平均梯度范数小于阈值，停止训练")
            self.stop = True
        if len(aloss) == 0:
            print("aloss {aloss},rand_loader :{len(rand_loader)}")
        losses.append(statistics.mean(aloss))
        update_time = time.time() - start_time
        print('Client:', self.cid, 'mean(aloss):', statistics.mean(aloss), '\tStd:', np.std(aloss), 'self.threshold:',
              self.threshold, '\tAAW epochs:', cnt)
        # print("after", self.AAweights[0])

        # for tensor1, tensor2 in zip(prev_AAweights, self.AAweights):
        #     if torch.allclose(tensor1, tensor2):
        #         print("newweight和初始的AAweights的元素相同")
        #     else:
        #         print("newweight和初始的AAweights的元素不同")

        # 计算更新所使用的时间
        update_time = time.time() - start_time
        self.timecost.append(update_time)
        #print("更新所使用的时间:", update_time, "秒")
        # print(f"client {self.cid} end   learning-----------------------")

        # print("Updated AAweight", type(self.AAweights))

        if statistics.mean(aloss) < 0.5:
            self.stop = True

    def adaptive_aggregation_weight_local_model(self, global_model: nn.Module, local_model: nn.Module) -> None:
        """
        Args:
            global_model: The received global/aggregated model.
            local_model: The trained local model.
        """
        # 创建local_model_t作为global_model的深拷贝
        uplocal_model = copy.deepcopy(global_model)
        # 获取local_model_t的参数列表,用来更新模型
        params_uplocal = list(uplocal_model.parameters())






        params_g = list(global_model.parameters())  # 获取全局模型的参数列表
        params_l = list(local_model.parameters())  # 获取本地模型的参数列表
        local_model_t = copy.deepcopy(global_model)
        params_l_t = list(local_model_t.parameters())

        global_model_t = copy.deepcopy(global_model)
        params_g_t = list(global_model_t.parameters())
        optimizer = torch.optim.SGD(params_g_t, lr=0)  # 使用全局模型的参数列表创建优化器

        init_value = self.sizerate  # 初始化权重的初始值

        if self.AAweights is None:
            self.AAweights = [torch.mul(torch.ones_like(param.data) ,init_value).to(self.device) for param in params_l]
            print("---------------------------------------------------------INit AAweight-shape is  -----------------------",self.AAweights [0].shape)

        self.updategrad=[(torch.ones_like(param.data) * 0).to(self.device) for param in params_g]
           # print("w", self.AAweights)
            # 如果是第一次更新，初始化 AAweight 为元素值为 init_value 的张量

        rand_ratio = self.rand_percent / 100  # 随机采样比例
        rand_num = int(rand_ratio * len(self.train_data))  # 计算随机采样的样本数量
        if rand_num < self.batch_size:
            print(
                f"ERROR: example rand_num={rand_num} < self.batch_size={self.batch_size} because len(self.train_data)={len(self.train_data)}, rand_ratio={rand_ratio}")
            return

        rand_idx = random.randint(0, len(self.train_data) - rand_num)  # 随机选择采样的起始索引
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=True)
        # 使用随机采样的数据创建 DataLoader

        losses = []  # 记录损失值
        cnt = 0  # 权重训练迭代计数器
        samples = 0
        print(f"---------------------------------------------------------client {self.cid} start  aaweight learning------------------------")


        # 记录更新前的时间戳
        start_time = time.time()
        #prev_AAweights = [aaweight.clone() for aaweight in self.AAweights]  # 复制之前的self.AAweights
        #print("before", self.AAweights[0])
        while True:
            aloss=[]
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output = global_model_t(x)  # 使用全局模型计算输出
                loss_value = self.loss(output, y)  # 计算损失值
                loss_value.backward()  # 反向传播计算梯度
                #此时可以获取梯度信息
                #print(f"y,{y},loss is:",loss_value.item())
                aloss.append(loss_value.item())
                #losses.append(loss_value.item())
                samples += 1
                #i=0
                for para_g, para_l_t, aaweight,upgrad,local_up in zip(params_g_t, params_l_t, self.AAweights,self.updategrad,params_uplocal):

                    #weight.data=weight-torch.mul(para_g.grad, self.eta * self.datasizerate) * para_l_t

                    aaweight.data = aaweight - torch.mul(para_g.grad, self.eta * self.sizerate) * para_l_t

                    #利用全局梯度对模型进行更新，防止过拟和，没有多次迭代
                    local_up.data -= 0.01 * para_g.grad.data

                    # print("self.eta * self.datasizerate", self.eta * self.sizerate )
                    # print("self.eta ", self.eta )
                    # print(" self.datasizerate", self.sizerate)
                    # print("para_g.grad", para_g.grad)
                    # print("para_l_t", para_l_t)
                    # if torch.equal(aaweight, updated_aaweight):
                    #     print("aaweight和更新后的AAweights的元素相同")
                    # else:
                    #     print("aaweight和更新后的AAweights的元素不同")
                    # self.AAweights[i] = updated_aaweight
                    # i=i+1

                    upgrad.data=upgrad+torch.mul(para_g.grad, self.eta * self.sizerate) * para_l_t
                # 更新 AAweight，根据全局模型在本地数据上的梯度和本地模型的参数

            losses.append(statistics.mean(aloss))
            cnt += 1
            if samples>=len(self.train_data):
                break
            # 只在后续迭代中训练一次权重

            # 当损失值的标准差小于阈值时，认为权重训练收敛
            if len(losses) > self.num_pre_loss and np.std(losses) < self.threshold:
                #print(f"self.num_pre_loss: {self.num_pre_loss},losses: {losses},samples: {samples}")
                #print('Client:', self.cid, '\tStd:', np.std(losses),'self.threshold:',self.threshold,
                #      '\tAAW epochs:', cnt)
                break

        #print("after", self.AAweights[0])

        # for tensor1, tensor2 in zip(prev_AAweights, self.AAweights):
        #     if torch.allclose(tensor1, tensor2):
        #         print("newweight和初始的AAweights的元素相同")
        #     else:
        #         print("newweight和初始的AAweights的元素不同")

        # 计算更新所使用的时间
        update_time = time.time() - start_time
        self.timecost.append(update_time)
        print("---------------------------------------------------------更新所使用的时间:", update_time, "秒")
        print(f"---------------------------------------------------------client {self.cid} end   learning-----------------------")

        #print("Updated AAweight", type(self.AAweights))





    def adaptive_aggregation_weight_ala(self,
                               global_model: nn.Module,
                               local_model: nn.Module) -> None:
        """
    Generates the Dataloader for the randomly sampled local training data and
    preserves the lower layers of the update.

    Args:
        global_model: The received global/aggregated model.
        local_model: The trained local model.

    Returns:
        None.
    """
        # print("adaptive_local_aggregation weight learning----------------")
    # randomly sample partial local training data


        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)

        self.rand_percent = 100
        self.batch_size = len(self.train_data)
        rand_idx = 0
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=True)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

    # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # # initialize the higher layers in the temp local model
        # for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
        #                                            self.weights):
        #     param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        self.updategrad = []
        sample = 0
        while True:

            for x, y in rand_loader:

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)  # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                       params_gp, self.AAweights):
                    # weight.data = torch.clamp(
                    # weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                    weight.data = weight - self.eta *0.01* (param_t.grad * (param_g - param))
                    if sample==0:
                        self.updategrad .append(param_t.grad)
                    else:
                        for ugrad,param_t in zip(self.updategrad, params_tp):
                            ugrad.data =ugrad+param_t.grad

                            # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                       params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight
                sample=sample+1

            losses.append(loss_value.item())
            cnt += 1


            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                  '\t AAW epochs:', cnt)
                break



    # obtain initialized local model
    # for param, param_t in zip(params_p, params_tp):
    #     param.data = param_t.data.clone()
    # print("ALA  weight type is",type(self.weights))





