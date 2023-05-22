import copy
import random
import time

import torch
from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.servers.serverbase import Server
from threading import Thread


class SCAFFOLD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        # 选择慢速客户端，并将其存储起来
        self.set_slow_clients()
        # 用于设置客户端，并根据参数 clientSCAFFOLD 来确定客户端的类型
        self.set_clients(args, clientSCAFFOLD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.server_learning_rate = args.server_learning_rate

        # 遍历 self.global_model 的参数，为每个参数创建了一个与其形状相同的全零张量，并将其添加到 self.global_c 列表
        # 为每个参数创建一个对应的共享参数，以便后续在聚合参数时进行更新和累积

        # 服务器控制变量，与模型参数形状相同
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            # 选择客户端，并将选中的客户端保存在self.selected_clients
            self.selected_clients = self.select_clients()
            self.send_models()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            # 接收客户端返回的更新模型。
            self.receive_models()
            # 根据接收到的模型更新，聚合全局模型的参数。
            self.aggregate_parameters()
            # 记录当前轮次的时间消耗，并将其保存在self.Budget列表
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # 如果启用了自动终止（由self.auto_break控制），并且满足终止条件（通过调用self.check_done()方法检查准确率等指标），则终止训练。
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()

    def send_models(self):
        # 将全局模型参数和共享参数发送给每个客户端，并记录发送操作的时间成本。
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model, self.global_c)
            # client.send_time_cost['num_rounds'] 表示客户端参与发送的轮数，在每轮发送操作前进行加一操作。
            client.send_time_cost['num_rounds'] += 1
            # 客户端的总发送时间成本，为什么*2
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):

        # 选择符合时间阈值的client
        # 如果客户端没有参与过任何轮次的训练或发送操作，那么将client_time_cost设置为
        # 0。如果client_time_cost小于等于指定的时间阈self.time_threthold，则将该客户端的ID添加到self.uploaded_ids列表中。

        assert (len(self.selected_clients) > 0)

        # random.sample()函数从self.selected_clients列表中随机选择指定int((1-self.client_drop_rate) * self.num_join_clients)数量的客户端作为活跃客户端，返回一个新的列表
        # active_clients。
        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))

        self.uploaded_ids = []
        # self.delta_ys = []
        # self.delta_cs = []
        for client in active_clients:
            # 计算每个客户端的平均训练时间成本 client_time_cost
            try:
                # client.train_time_cost['total_cost'] 表示客户端的总训练时间成本
                # client.train_time_cost['num_rounds'] 表示客户端参与训练的轮数
                # client.send_time_cost['total_cost'] 表示客户端的总发送时间成本
                # client.send_time_cost['num_rounds'] 表示客户端参与发送的轮数
                # client_time_cost 计算为平均每一轮的训练时间成本和平均每一轮发送时间成本之和。
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                # self.delta_ys.append(client.delta_y)
                # self.delta_cs.append(client.delta_c)

    def aggregate_parameters(self):

        # ‘’‘
        # 首先，创建两个变量
        # global_model
        # 和
        # global_c，分别用于存储全局模型参数和全局聚合参数。
        # 遍历已上传参数的客户端标识符列表
        # self.uploaded_ids。
        # 对于每个客户端标识符
        # cid，获取该客户端的
        # delta_yc()，其中返回了该客户端的模型参数更新量。
        # 遍历全局模型参数和客户端参数更新量，将客户端参数更新量按比例累加到全局模型参数中，计算公式为
        # server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate。
        # 同样地，遍历全局聚合参数和客户端参数更新量，将客户端参数更新量按比例累加到全局聚合参数中，计算公式为
        # server_param.data += client_param.data.clone() / self.num_clients。
        # 将更新后的全局模型参数赋值给
        # self.global_model，将更新后的全局聚合参数赋值给
        # self.global_c。
        # 这段代码通过对客户端上传的参数进行累加，实现了参数的聚合操作。在聚合过程中，每个客户端的参数更新量被按比例加到全局模型参数和全局聚合参数上，从而得到了更新后的全局模型参数和全局聚合参数
        # ’‘’
        # original version
        # for dy, dc in zip(self.delta_ys, self.delta_cs):
        #     for server_param, client_param in zip(self.global_model.parameters(), dy):
        #         server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
        #     for server_param, client_param in zip(self.global_c, dc):
        #         server_param.data += client_param.data.clone() / self.num_clients

        # save GPU memory
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for cid in self.uploaded_ids:
            dy, dc = self.clients[cid].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.num_clients

        self.global_model = global_model
        self.global_c = global_c