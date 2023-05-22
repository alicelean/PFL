import time,os
from flcore.clients.clientala import clientALA
from flcore.servers.serverbase import Server
from threading import Thread
mpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Programpath = "/".join(mpath.split("/")[:-1])
print(mpath,Programpath)

import pandas as pd
class FedALA(Server):
    def __init__(self, args, times,filedir):
        super().__init__(args, times,filedir)
        # select slow clients
        self.set_slow_clients()

        self.method = "FedALA"

        self.set_clients(clientALA)




        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []



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
            print(f"INFO:-----------------global_rounds is---{i}-----,select client num is-----{len(select_id)}-select id :--{ids}----------------------------------")
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
            self.set_new_clients(clientALA)
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
