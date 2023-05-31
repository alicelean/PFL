import torch
import copy
import random
import time
from flcore.clients.clientapple import clientAPPLE
from flcore.servers.serverbase import Server
from threading import Thread
from utils.dlg import DLG
from utils.data_utils import read_client_data
import pandas as pd


class APPLE(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        self.fix_ids = True
        self.method = "APPLE"

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAPPLE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.fix_ids = True
        self.method = "APPLE"

        self.client_models = [copy.deepcopy(c.model_c) for c in self.clients]

        train_samples = 0
        for client in self.clients:
            train_samples += client.train_samples
        p0 = [client.train_samples / train_samples for client in self.clients]

        for c in self.clients:
            c.p0 = p0


    def train(self):
        print("*************************** server_aaw train ***************************")
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
                print("\nEvaluate personalized models")
                res = self.evaluate(i)
                resc = [self.filedir]
                for line in res:
                    resc.append(line)
                colum_value.append(resc)

            for client in self.clients:
                client.train(i)

            # threads = [Thread(target=client.train)
            #            for client in self.clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.receive_models()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

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
        # -----------------------------------------------------------------------------------------------

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAPPLE)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        self.args.num_clients = self.num_clients
        

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_models(self.client_models)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.num_join_clients))

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
                self.uploaded_models.append(client.model_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model_server in zip(range(self.num_clients), self.client_models):
            client_model = self.clients[cid].model
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(client_model_server.parameters(), client_model.parameters()):
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
        self.args.num_clients = self.num_clients + self.num_new_clients
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
        self.client_models += [copy.deepcopy(c.model_c) for c in self.new_clients]

        train_samples = 0
        for client in self.clients + self.new_clients:
            train_samples += client.train_samples
        p0 = [client.train_samples / train_samples for client in self.clients + self.new_clients]

        for c in self.clients + self.new_clients:
            c.p0 = p0
            
        for client in self.new_clients:
            client.set_models(self.client_models)
            for e in range(self.fine_tuning_epoch):
                client.train(self.global_rounds)
