import time,os
from flcore.clients.client_alajs import clientALAJS
from flcore.servers.serverbase import Server
from utils.distance import jensen_shannon_distance
from utils.data_utils1 import read_client_data_new
from threading import Thread
mpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Programpath = "/".join(mpath.split("/")[:-1])
print(mpath,Programpath)
import random

import pandas as pd
class ALAJS(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        # select slow clients
        self.set_slow_clients()

        #设置client数据
        self.client_batch_size = args.batch_size
        self.client_learning_rate = args.local_learning_rate
        self.client_local_epochs = args.local_epochs
        self.set_clients(clientALAJS)
        self.fix_ids=False



        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.method="ALAJS"


    def train(self):
        if self.fix_ids:
            self.select_idlist = self.read_selectInfo()

        colum_value = []
        # -----------写入超参数
        #1:minist,
        redf = pd.DataFrame(columns=["dataset","global_rounds","client_enpoches","client_batch_size","client_learning_rate","ratio", "client_num", "Dirichlet alpha"])
        redf.loc[len(redf) + 1] = ["dataset","global_rounds","client_enpoches","client_batch_size","client_learning_rate","ratio", "client_num", "Dirichlet alpha"]
        redf.loc[len(redf) + 1] = [self.filedir,self.global_rounds,self.client_local_epochs,self.client_batch_size,self.client_learning_rate,self.join_ratio, self.num_clients, 0.1]
        path = Programpath+"/res/"+self.method+"/canshu.csv"
        redf.to_csv(path, mode='a', header=False)
        # ---------------------------------------
        select_id=[]
        for i in range(self.global_rounds+1):
            s_t = time.time()
            #----设置不同的选择方式
            if self.fix_ids:
                self.selected_clients = self.select_clients(self.global_rounds)
            else:
                #随机选择
                self.selected_clients = self.select_clients()


            # ----------------------写入每次选择的client的数据-----------------------
            ids = []
            for client in self.selected_clients:
                ids.append(client.id)
                # clientvalue.append(
                #     [rouds, client.id, client.client_label, client.train_samples, client.test_samples, client.distance,
                #      client.local_steps, client.learning_rate, client.last_loss])
            print(f"INFO:-----------------global_rounds is---{i}-----,select client num is-----{len(ids)}-select id :--{ids}----------------------------------")
            select_id.append([i, ids])
            #---------------------------------------------------------------------



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

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 写入idlist----保证整个客户端选择一致，螚更好的判别两种算法的差异---------------------------------------
        redf = pd.DataFrame(columns=["global_rounds", "id_list"])
        redf.loc[len(redf) + 1] = ["global_rounds", "id_list"]
        for v in range(len(select_id)):
            redf.loc[len(redf) + 1] = select_id[v]
        idpath = Programpath+"/res/selectids/select_client_ids" + str(self.num_clients) + "_" + str(self.join_ratio) + ".csv"
        redf.to_csv(idpath, mode='a', header=False)

        # --------------训练过程中的全局模型的acc
        colum_name = ["case", "method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC"]
        redf = pd.DataFrame(columns=colum_name)
        redf.loc[len(redf) + 1] = colum_name
        for i in range(len(colum_value)):
            redf.loc[len(redf) + 1] = colum_value[i]
        path = Programpath+"/res/"+self.method+"/acuuray.csv"
        redf.to_csv(path, mode='a', header=False)

        #-----------------------------------------------------------------------------------------------

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))




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
            client.local_initialization(self.global_model)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.local_initialization(self.global_model)
            for e in range(self.fine_tuning_epoch):
                client.train()


    def set_clients(self, clientObj):
        # 整个数据集的标签向量
        alllabelvectors = []
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data, train_label = read_client_data_new(self.dataset, i, self.filedir, is_train=True)
            test_data, test_label = read_client_data_new(self.dataset, i, self.filedir, is_train=False)
            # 单个client的标签向量----------
            client_label = train_label
            for j in range(len(test_label)):
                client_label[j] = test_label[j] + train_label[j]
            if i == 0:
                alllabelvectors = client_label
            else:
                for j in range(len(client_label)):
                    alllabelvectors[j] += client_label[j]

            # 单个client的标签向量----------
            client = clientObj(self.args,
                            id=i,
                            filedir=self.filedir,
                            client_label=client_label,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)
        # ------------计算client与总体的差异距离---------------------------------
        alldistance = 0
        for c in self.clients:
                # ---------
            c.distance = jensen_shannon_distance(c.client_label, alllabelvectors)
            alldistance += c.distance
            print(f"INFormation-----------------------------------init client {c.id} JS distance is {c.distance}-------------------------------")
        for c in self.clients:
            c.alldistance = alldistance



        print(f"Info:***********************************ALAJS set_clients**************************")

        # 读完数据后写入--------------------
        # redf = pd.DataFrame(columns=dataname)
        # redf.loc[len(redf) + 1] = dataname
        # redf.loc[len(redf) + 1] = traindatalength
        # redf.loc[len(redf) + 1] = testdatalength
        # path = Programpath+"/res/clients.csv"
        # redf.to_csv(path, mode='a', header=False)
        # ---------------------------------------
        # # 所有client的数据信息
        # redf = pd.DataFrame(columns=labelname)
        # redf.loc[len(redf) + 1] = labelname
        # for value in labelvalue:
        #     redf.loc[len(redf) + 1] = value
        # print("INFormation--labelpath -", self.num_clients, self.join_ratio, self.filedir, self.method)
        # labelpath = Programpath+"/res/" + self.method + "/" + "allclients" + str(self.num_clients) + "_" + str(self.join_ratio) + ".csv"
        # redf.to_csv(labelpath, mode='a', header=False)


    def receive_models(self):
        print(f"Info:***********************************ALAJS receive_models**************************")
        assert (len(self.selected_clients) > 0)
       #随机选择一些激活一些client
        active_clients = random.sample(self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))
        active_clients=self.selected_clients

        active_train_samples = 0
        for client in active_clients:
            active_train_samples += client.train_samples

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(client.model)

                self.uploaded_weights.append(client.train_samples)
                a = 0.5
                # print(f"Information -------------aggregation method is {self.method}-------------")
                weight = a * (client.train_samples / active_train_samples) / (1 - a) * (
                        client.distance / client.alldistance)




        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
