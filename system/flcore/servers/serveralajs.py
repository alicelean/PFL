import time,os
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.clients.clientalajs import clientALAJS
from utils.data_utils import read_client_data
from utils.distance import jensen_shannon_distance
import pandas as pd
import numpy as np
import math
import random
class ALAJS(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        # select slow clients
        self.set_slow_clients()

        self.method = "ALAJS"

        self.set_clients(clientALAJS)
        self.fix_ids = True




        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("------------------Finished creating server and clients.----------------------")

        # self.load_model()
        self.Budget = []



    def train(self):
        print("*************************** serveralajs train ***************************")
        self.writeparameters()
        colum_value = []
        select_id = []

        if self.fix_ids:
            file_path = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(
                self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            self.select_idlist = self.read_selectInfo(file_path)


        for i in range(self.global_rounds+1):
            s_t = time.time()

            # ----2.设置不同的选择方式---------------------------------------------------
            if self.fix_ids:
                self.selected_clients = self.select_clients(i)
            else:
                self.selected_clients = self.select_clients()
                # ----------------------3.写入每次选择的client的数据----------------------------
                ids = []
                for client in self.selected_clients:
                    ids.append(client.id)
                select_id.append([i, ids])
                # -------------------------------------------------------------------------

            # -------------------------------------------------------------------------




            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                res=self.evaluate(i)
                #记录当前模型的状态，loss,accuracy等
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
            print("\nreceive_models")
            self.receive_models()

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

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

        if self.num_new_clients > 0:
            print(f" self.num_new_clients is ----{ self.num_new_clients }----")
            self.eval_new_clients = True
            self.set_new_clients(clientALAJS)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

            #client.local_initialization(self.global_model)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.local_initialization(self.global_model)
            for e in range(self.fine_tuning_epoch):
                client.train()

    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples += len(train_data) + len(test_data)
            client = clientObj(self.args,
                               id=i,
                               traindata = train_data,
                                testsdata = test_data,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()
            client.sizerate = (client.train_samples + client.test_samples) / samples
        #根据label 来计算distance
        self.setdistance()

            # print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()

    def setdistance(self):
        alllabel = [0 for i in range (10)]
        for client in self.clients:
            #print(f"before alllabel is {alllabel},client.label is{client.label}")
            for j in range(10):
                alllabel[j]+=client.label[j]
            #print(f"after alllabel is {alllabel}")
        alldistance = 0
        for client in self.clients:
            #client.distance = jensen_shannon_distance(client.label, alllabel)
            client.distance = self.calculate_hellinger_distance(client.label, alllabel)
            alldistance += client.distance
            print(f"INFormation-------------serveralajs----------------------init client {client.id} JS distance is {client.distance}-------------------------------")

        for client in self.clients:
            client.alldistance = alldistance
            client.alllabel = alllabel
            print(f"set clients info : client {client.id},distance rate is {client.distance/client.alldistance},sizerate:{client.sizerate}")



    def receive_models(self):
        '''
        根据设定的客户端丢失率、时间阈值和客户端的训练时间消耗，
        选择符合条件的活跃客户端，并收集其模型和样本权重。最后，对样本权重进行归一化，以便后续在联邦学习中使用。
        Returns:

        '''
        assert (len(self.selected_clients) > 0)
        active_clients=self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        active_train_samples=0
        for client in active_clients:
            active_train_samples+=client.train_samples
        active_distance = 0
        # for client in active_clients:
        #     active_distance += client.distance

        activelabel=[0 for i in range(10)]
        for client in active_clients:
            for j in range(10):
                activelabel[j] += client.label[j]

        for client in active_clients:
            client.distance =jensen_shannon_distance(client.label, activelabel)
            #client.distance = self.calculate_hellinger_distance(client.label, activelabel)
            active_distance += client.distance



        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:

                #a=0.5
                # if a == 0:
                #     jsweight = (1 / (client.distance / client.alldistance))
                # elif a==1:
                #     jsweight = client.train_samples / active_train_samples
                # else:
                #     distanceweight = (1 / (client.distance / active_distance))*(1-a)
                #     jsweight=distanceweight*a * (client.train_samples / active_train_samples)
                #print(f"origin weight is {client.train_samples / active_train_samples},jsweight is {jsweight}")

                #1.origin datasize

                #jsweight = client.train_samples / active_train_samples
                #2. distance,目前只选择这个定义权重的公式
                scale_factor = 0.1
                jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor) + 0.5 * (
                            client.train_samples / active_train_samples)

                #jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor)
                #3.
                #jsweight=self.weight_from_distance1(client.distance / active_distance, scale_factor)

                #jsweight=self.weight_from_distance3( client.distance / active_distance, scale_factor, client.train_samples / active_train_samples)

                #+0.5 * (client.train_samples / active_train_samples)

               # jsweight=self.weight_from_distance3(client.distance / client.alldistance, scale_factor, client.train_samples / active_train_samples)


                print(f"origin weight is {client.train_samples / active_train_samples},jsweight is {jsweight}")


                self.uploaded_weights.append(jsweight)
                self.uploaded_models.append(client.model)
                self.uploaded_ids.append(client.id)
        print("before guiyihua",self.uploaded_weights)
        self.guiyihua_weight()
        print("after guiyihua", self.uploaded_weights)



    def guiyihua_weight(self):

        #weights_array = np.array(self.uploaded_weights)
        total_sum = np.sum(self.uploaded_weights)
        self.uploaded_weights = [i/total_sum for i in self.uploaded_weights ]


    def weight_from_distance2(self,distance, scale_factor=0.1, exponent=0.5):
        '''
        distance：表示数据分布的距离。
        scale_factor：用于调整权重的比例因子。
        exponent：用于控制指数函数的指数。
        确定合适的scale_factor和exponent值是根据具体问题和数据分布而定的，没有固定的通用值。你可以根据实际情况进行实验和调整，以找到适合你问题的最佳值。

一般来说，scale_factor的选择可以考虑数据的范围和分布。如果数据的范围很大，可以选择较大的scale_factor，以使权重下降得更快。如果数据的范围较小，可以选择较小的scale_factor，以使权重下降得更慢。

对于exponent，它控制指数函数的指数，可以调整权重下降的速率和曲线的形状。较大的exponent会使权重下降得更快，而较小的exponent会使权重下降得更慢。你可以根据问题的需求和期望的权重分布形状进行调整。

建议尝试不同的值并观察权重的变化和模型的表现。通过反复实验和调整，你可以找到适合你问题的最佳scale_factor和exponent值。
        Args:
            scale_factor:
            exponent:

        Returns:

        '''
        if distance == 0:
            return 1.0
        else:
            return math.exp(-exponent * distance * scale_factor)

    def weight_from_distance3(self,distance, scale_factor, data_size):
        '''
                  其中：
            distance 表示数据点的距离。
            scale_factor 是一个比例因子，用于调整 Sigmoid 函数的斜率。
            data_size 表示数据集的大小，影响 Sigmoid 函数的输入范围。
            通过调整 scale_factor 的值，你可以控制 Sigmoid 函数的斜率和权重的变化速率。而数据集的大小 data_size 可以影响 Sigmoid 函数的输入范围，进而影响权重的分布。
            Args:
                distance:
                scale_factor:
                data_size:

            Returns:
                Args:
                    distance:
                    scale_factor:
                    data_size:

                Returns:

        '''


        sigmoid_input = scale_factor * (data_size / 2 - distance)
        weight = 1 / (1 + math.exp(sigmoid_input))


        return weight


   #不同的距离计算

    def calculate_kl_divergence(self,p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        kl_divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return kl_divergence

    def calculate_hellinger_distance(self,p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        sqrt_p = np.sqrt(p)
        sqrt_q = np.sqrt(q)
        hellinger_distance = np.sqrt(np.sum((sqrt_p - sqrt_q) ** 2)) / np.sqrt(2)
        return hellinger_distance

    # # 示例数据
    # label1 = [10, 15, 5, 20]  # 第一个标签的样本个数分布
    # label2 = [5, 10, 10, 25]  # 第二个标签的样本个数分布
    #
    # # 计算 KL 散度
    # kl_divergence = calculate_kl_divergence(label1, label2)
    # print("KL 散度:", kl_divergence)
    #
    # # 计算 Hellinger 距离
    # hellinger_distance = calculate_hellinger_distance(label1, label2)
    # print("Hellinger 距离:", hellinger_distance)











