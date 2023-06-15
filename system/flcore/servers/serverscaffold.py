import copy,os
import random
import time

import torch
from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.servers.serverbase import Server
from threading import Thread
mpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Programpath = "/".join(mpath.split("/")[:-1])
print(mpath,Programpath)

import pandas as pd

class SCAFFOLD(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        self.fix_ids=True
        self.method = "SCAFFOLD"

        # select slow clients
        # 选择慢速客户端，并将其存储起来
        self.set_slow_clients()
        # 用于设置客户端，并根据参数 clientSCAFFOLD 来确定客户端的类型
        #self.set_clients(args, clientSCAFFOLD)
        self.set_clients(clientSCAFFOLD)

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
        print("*************************** server_sccafoold train ***************************")
        self.writeparameters()
        colum_value = []
        select_id = []

        if self.fix_ids:
            file_path = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            self.select_idlist = self.read_selectInfo(file_path)
        # -----------写入超参数
        # 1:minist,
        redf = pd.DataFrame(
            columns=["dataset", "global_rounds", "client_enpoches", "client_batch_size", "client_learning_rate",
                     "ratio", "client_num", "Dirichlet alpha"])
        redf.loc[len(redf) + 1] = ["dataset", "global_rounds", "client_enpoches", "client_batch_size",
                                   "client_learning_rate", "ratio", "client_num", "Dirichlet alpha"]
        redf.loc[len(redf) + 1] = [self.filedir, self.global_rounds, self.client_local_epochs, self.client_batch_size,
                                   self.client_learning_rate, self.join_ratio, self.num_clients, 0.1]
        path = Programpath + "/res/" + self.method + "/canshu.csv"
        redf.to_csv(path, mode='a', header=False)
        # ---------------------------------------
        select_id = []
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            # 选择客户端，并将选中的客户端保存在self.selected_clients
            # ----设置不同的选择方式
            if self.fix_ids:
                self.selected_clients = self.select_clients(self.global_rounds)
            else:
                # 随机选择
                self.selected_clients = self.select_clients()
            # ----------------------写入每次选择的client的数据-----------------------
            ids = []
            for client in self.selected_clients:
                ids.append(client.id)
                    # clientvalue.append(
                    #     [rouds, client.id, client.client_label, client.train_samples, client.test_samples, client.distance,
                    #      client.local_steps, client.learning_rate, client.last_loss])
            print(
                    f"INFO:-----------------global_rounds is---{i}-----,select client num is-----{len(select_id)}-select id :--{ids}----------------------------------")
            select_id.append([i, ids])
                # ---------------------------------------------------------------------


            self.send_models()
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                res = self.evaluate(i)
                # 记录当前模型的状态，loss,accuracy等
                resc = [self.filedir]
                for line in res:
                    resc.append(line)
                colum_value.append(resc)
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
        # 6.写入idlist----保证整个客户端选择一致，更好的判别两种算法的差异---------------------------------------
        if not self.fix_ids:
            redf = pd.DataFrame(columns=["global_rounds", "id_list"])
            redf.loc[len(redf) + 1] = ["*********************", "*********************"]
            redf.loc[len(redf) + 1] = ["global_rounds", "id_list"]
            for v in range(len(select_id)):
                redf.loc[len(redf) + 1] = select_id[v]
            idpath = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            redf.to_csv(idpath, mode='a', header=False)
            print("write select id list ", idpath)
        # --------------7.训练过程中的全局模型的acc
        colum_name = ["case", "method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC"]
        redf = pd.DataFrame(columns=colum_name)
        redf.loc[len(redf) + 1] = colum_name
        for i in range(len(colum_value)):
            redf.loc[len(redf) + 1] = colum_value[i]
        accpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_acc.csv"
        print("success training write acc txt", accpath)
        redf.to_csv(accpath, mode='a', header=False)
        print(colum_value)
        # ---------------记录数据-----------
        redf = pd.DataFrame(columns=["dataset", "method", "round", "ratio", "average_time_per", "time_list"])
        redf.loc[len(redf) + 1] = [self.dataset, self.method, self.global_rounds, self.join_ratio,
                                   sum(self.Budget[1:]) / len(self.Budget[1:]), self.Budget[1:]]
        accpath = self.programpath + "/res/time_cost.csv"
        print("success training write acc txt", accpath)
        redf.to_csv(accpath, mode='a', header=False)
        # -----------------------------------------------------------------------------------------------
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