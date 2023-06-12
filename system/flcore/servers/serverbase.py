import torch
import os
import numpy as np
import h5py
import copy
import time,math
import random
import pandas as pd
from utils.distance import jensen_shannon_distance
from utils.data_utils import read_client_data
from utils.dlg import DLG
import ast
# Programpath="/Users/alice/Desktop/python/PFL/"
# mpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Programpath = "/".join(mpath.split("/")[:-1])
# print("serverbase",mpath,Programpath)

from utils.vars import Programpath

#Programpath=" /home/alice/Desktop/python/PFL/"
class Server(object):
    def __init__(self, args, times,filedir="test"):
        #记录数据文件的位置，同一个数据集不同程度的数据分布差异
        self.filedir=args.dataset
        self.method =None
        self.select_idlist = []
        self.randomSelect = False
        # 设置client数据
        self.client_batch_size = args.batch_size
        self.client_learning_rate = args.local_learning_rate
        self.client_local_epochs = args.local_epochs
        self.fix_ids = False
        self.ISAAW=False
        self.programpath=Programpath

        # 记录所有的weights
        self.allweights = []




        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

    def set_clients_origin(self, clientObj):
        print("**************************1.INfo,serverbase set_clients ,init data********************")
        samples=0
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            samples+=len(train_data)+len(test_data)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

        for client in self.clients:
            client.setlabel()
            client.sizerate=(client.train_samples+client.test_samples)/samples

            #print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self,round=-1):
        if round==-1:
            # 随机选择client
            if self.random_join_ratio:
                num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            else:
                num_join_clients = self.num_join_clients
            selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))

            return selected_clients
        else:
            # 按照制定方式选择clients,self.select_idlist[group],设定每一轮要选择的client,方便对不同算法进行比较
            selected_clients = []
            ids = []
            for c in self.clients:
                if c.id in self.select_idlist[round]:
                    selected_clients.append(c)
                    ids.append(c.id)
            print(f"INFO:----------fix client id is :group is {round} select id list is:", ids, type(selected_clients[0]))
            return selected_clients


    def send_models(self):
        '''
        将globalmodel复制给本地模型,并记录time cost
        Returns:

        '''
        assert (len(self.clients) > 0)

        # 更新模型参数，将globalmodel复制给本地模型
        for client in self.selected_clients:
            if self.ISAAW:
                client.local_initialization(self.global_model, round)


        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        '''
        根据设定的客户端丢失率、时间阈值和客户端的训练时间消耗，
        选择符合条件的活跃客户端，并收集其模型和样本权重。最后，对样本权重进行归一化，以便后续在联邦学习中使用。
        Returns:

        '''
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))
        active_clients=self.selected_clients

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
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples















    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w



    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, group, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        try:
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_acc = 0.0

        try:
            test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        except ZeroDivisionError:
            test_auc = 0.0

        try:
            train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        except ZeroDivisionError:
            train_loss = 0.0

        try:
            accs = [a / n for a, n in zip(stats[2], stats[1])]
        except ZeroDivisionError:
            accs = [0.0] * len(stats[2])

        try:
            aucs = [a / n for a, n in zip(stats[3], stats[1])]
        except ZeroDivisionError:
            aucs = [0.0] * len(stats[3])

        # test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        # test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        # accs = [a / n for a, n in zip(stats[2], stats[1])]
        # aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return [self.method, group, train_loss, test_acc, test_auc, np.std(accs), np.std(aucs)]



    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        #增加新的client
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            #将全局模型参数复制给client
            client.set_parameters(self.global_model)
            for e in range(self.fine_tuning_epoch):
                print(f"fine_tuning_epoch is {fine_tuning_epoch}")
                client.train()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    #add
    def read_selectInfo(self,file_path):
        print("INFO:---------Using fix select id list--------------------------------")

        data = pd.read_csv(file_path)
        print(file_path)
        print(data)
        rounds = data['global_rounds'].tolist()
        # print(data['ids'].tolist()[0])
        ids = [ast.literal_eval(i) for i in data['id_list'].tolist()]
        # print("rounds",rounds)
        # print("ids",ids)
        id_dict = {}
        for i in range(len(rounds)):
            id_dict[rounds[i]] = ids[i]
        print(id_dict)
        return id_dict

    def evaluate_origin(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
    def writeparameters(self):
       # -----------1.写入超参数
       # 1:minist,
       redf = pd.DataFrame(
           columns=["dataset", "global_rounds", "client_enpoches", "client_batch_size", "client_learning_rate", "ratio",
                    "client_num", "Dirichlet alpha"])
       redf.loc[len(redf) + 1] = ["dataset", "global_rounds", "client_enpoches", "client_batch_size",
                                  "client_learning_rate", "ratio", "client_num", "Dirichlet alpha"]
       redf.loc[len(redf) + 1] = [self.dataset, self.global_rounds, self.client_local_epochs, self.client_batch_size,
                                  self.client_learning_rate, self.join_ratio, self.num_clients, 0.1]
       path = self.programpath + "/res/" + self.method + "/" + self.dataset + "_canshu.csv"
       redf.to_csv(path, mode='a', header=False)
       print("-------------------------------------------------------------------------write exe canshu,txt path is",path)
       # ---------------------------------------

    def writeclientInfo(self):
        '''
        写入设置的clients数据信息
        Returns:

        '''
        clientinfo = []
        dataname = ["id", "train_len", "test_len","sizerate", "train_slow", "send_slow","label"]
        path = self.programpath + "res/" + self.method + "/clientsInfo.csv"
        for c in self.clients:
            clientinfo.append([c.id, c.train_samples, c.test_samples,c.sizerate, c.train_slow, c.send_slow,c.label])
        redf = pd.DataFrame(columns=dataname)
        for value in clientinfo:
            redf.loc[len(redf) + 1] = value
        redf.to_csv(path, mode='a', header=False)

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
            client.sizerate = (client.train_samples + client.test_samples) / samples
        # 根据label 来计算distance
        # self.setdistance()

        # print(f"client {client.id} ,sizerate is {client.sizerate}")
        self.writeclientInfo()

    def select_weight_vector(self):
        active_distance = 0
        activelabel = [0 for i in range(10)]
        active_train_samples = 0

        for client in self.selected_clients:
            active_train_samples += client.train_samples
            for j in range(10):
                activelabel[j] += client.label[j]
        for client in self.selected_clients:
            # 计算参与训练的client分布差异
            client.distance = jensen_shannon_distance(client.label, activelabel)
            active_distance += client.distance

        for client in self.selected_clients:
            scale_factor = 0.1
            jsweight = self.weight_from_distance2(client.distance / active_distance, scale_factor) + 0.5 * (
                    client.train_samples / active_train_samples)

            client.AAW.initweight = jsweight
            client.AAW.initweight = client.train_samples / active_train_samples
            # 第一次参与训练需要初始化它的weight，weight 按照js计算weight的方式进行
            if not client.AAW.hasinit:
                client.AAW.init_weight(self.global_model)
                client.AAW.hasinit = True


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


