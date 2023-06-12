import time
from flcore.clients.client_newaaw import clientNewAAW
from flcore.servers.serverbase import Server
from threading import Thread
import pandas as pd
from utils.distance import jensen_shannon_distance
from utils.data_utils import read_client_data
import torch
class NewAAW(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        self.method="NewAAW"
        self.fix_ids = True
        self.eta=0.001



        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientNewAAW)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):

        self.writeparameters()
        colum_value = []
        select_id = []

        if self.fix_ids:
            file_path = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(self.num_clients) + "_" + str(self.join_ratio) + ".csv"
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

            #-------------------------------------------------------------------------
            #给没有初始化权重的client 初始化权重
            self.select_weight_vector()

            #将 global model 发送给client
            self.send_models(i)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}---Evaluate global model----------")
                res = self.evaluate(i)
                # --------------------4.记录当前模型的状态，loss,accuracy----------------------
                resc = [self.filedir]
                for line in res:
                    resc.append(line)
                resc.append(self.join_ratio)
                colum_value.append(resc)
                print("envaluate:",i,colum_value)
                # -------------------------------------------------------------------------

            for client in self.selected_clients:
                #记录client 的全体梯度
                if client.eta>0:
                    client.get_prev_grads()
                if client.update:
                    client.updateweight()
                client.train()


            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models(i)

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
            idpath = self.programpath + "/res/selectids/" + self.dataset + "_select_client_ids" + str(self.num_clients) + "_" + str(self.join_ratio) + ".csv"
            redf.to_csv(idpath, mode='a', header=False)
            print("write select id list ",idpath)
        # --------------7.训练过程中的全局模型的acc
        colum_name = ["case", "method", "group", "Loss", "Accurancy", "AUC", "Std Test Accurancy", "Std Test AUC","join_ratio"]
        redf = pd.DataFrame(columns=colum_name)
        redf.loc[len(redf) + 1] = colum_name
        for i in range(len(colum_value)):
            redf.loc[len(redf) + 1] = colum_value[i]
        accpath = self.programpath + "/res/" + self.method + "/" + self.dataset + "_acc.csv"
        print("success training write acc txt",accpath)
        redf.to_csv(accpath, mode='a', header=False)
        print(colum_value)
        # -----------------------------------------------------------------------------------------------






        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientNewAAW)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def select_weight_vector(self):
        active_distance = 0
        activelabel = [0 for i in range(10)]
        active_train_samples = 0

        for client in self.selected_clients:
            active_train_samples += client.train_samples
            for j in range(10):
                activelabel[j] += client.label[j]
        for client in self.selected_clients:
            #计算参与训练的client分布差异
            client.distance = jensen_shannon_distance(client.label, activelabel)
            active_distance += client.distance

        for client in self.selected_clients:
            # scale_factor = 0.1
            # jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor) + 0.5 * (
            #                 client.train_samples / active_train_samples)
            #
            # client.AAW.initweight =jsweight

            #第一次参与训练需要初始化它的weight，weight 按照js计算weight的方式进行
            if not client.hasinit:
                client.initweight = client.train_samples / active_train_samples
                client.init_weight(self.global_model)
                client.hasinit=True
    def add_parameters(self, weight, client_model):
        '''
        使用张量权重聚合模型
        Args:
            aaw:
            client_model:

        Returns:

        '''
        #print("aaw aggregator --->global model",type(aaw),type(self.global_model.parameters()),type(client_model.parameters()))
        parmes_g=self.global_model.parameters()
        parmes_l=client_model.parameters()
        print(type(weight))
        for w,server_param, client_param in zip(weight,parmes_g,parmes_l):
            if torch.is_tensor(w) and torch.is_tensor(client_param.data) and w.shape == client_param.data.shape:
                server_param.data += torch.mul(client_param.data.clone(), w)
            else:
                print(f"Error: serveraaw add_parameters Invalid tensor shape or type,w.shape {w.shape} but client_param.data.shape {client_param.data.shape}")

    def set_clients(self, clientObj):
        print("**************************1.INfo,set_clients ,init data********************")
        samples = 0
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples += len(train_data) + len(test_data)
            client = clientObj(self.args,
                               id=i,
                               traindata=train_data,
                               testsdata=test_data,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()

    def send_models(self,round):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)
        #记录没开始的本地模型
        if round==0:
            for client in self.clients:
                client.init_prev_params(self.global_model)
                #print(f"round is {round}, client {client.id} init prev params is {client.prev_params}")
        # 更新模型参数，将globalmodel复制给本地模型
        # for client in self.selected_clients:
        #     if self.ISAAW:
        #         client.local_initialization(self.global_model, round)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def receive_models(self,round):
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

        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:


                self.uploaded_weights.append(client.AAweight)
                #print("aggregation weight is :",type(client.AAweight))
                self.uploaded_models.append(client.model)
                self.uploaded_ids.append(client.id)

        self.allweights.append([round, self.uploaded_weights])
        #print("before guiyihua",self.uploaded_weights)
        #self.guiyihua_weight()
        self.normalize_update_weight()
        #print("after guiyihua", self.uploaded_weights)

    def normalize_update_weight(self):
        '''
        tensor list进行归一化
        Returns:

        '''

        #将每一个权重对应位置的tensor进行归一化，
        tensorlists=[]
        for i in range(len(self.uploaded_weights[0])):
            tensorlists.append([])
        for weight in self.uploaded_weights:
            for j in range(len(weight)):
                tensorlists[j].append( weight[j])

        for j in range(len(tensorlists)):
            tensor_list=tensorlists[j]
            tensor_sum = sum(tensor_list)
            normalized_list = [tensor / tensor_sum for tensor in tensor_list]
            tensorlists[j]=normalized_list

        #更新self.uploaded_weights
        self.uploaded_weights=[]
        for i in range(len(tensorlists[0])):
            self.uploaded_weights.append([0 for j in range(len(tensorlists))])


        for i in range(len(tensorlists)):
            tensor_list=tensorlists[i]
            for j in range(len(tensor_list)):
                self.uploaded_weights[j][i]=tensor_list[j]
