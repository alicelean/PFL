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
        self.fix_ids = False



        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []



    def train(self):
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
        # -----------------------------------------------------------------------------------------------

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
