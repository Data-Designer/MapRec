#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 10:52
# @Author  : Jack Zhao
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm

# #Desc: trainer
import faiss
import torch
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import ConcatDataset

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from config import config
from utils import get_history, calculate_metrics, tf_idf, bpr_dataset
from tensorflow import keras
from models.base import MFBasedModel, DNNBasedModel, GMFBasedModel, MFBasedModelTwo, MFBaseModelBPR,GMFBasedModelTwo,DNNBasedModelTwo


class MetaRecTrainer():
    def __init__(self, config):
        super(MetaRecTrainer, self).__init__()
        # base_info
        self.use_cuda = config['USE_CUDA']
        self.base_model = config['BASE_MODEL']
        self.root = config['ROOT']
        self.ratio = config['RATIO']
        self.task = config['TASK']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src
        # hyper
        self.epoch = config['EPOCH']
        self.emb_dim = config['EMB_DIM']
        self.meta_dim = config['META_DIM']
        self.num_fields = config['NUM_FIELDS']
        self.lr = config['LR']
        self.wd = config['WD']
        self.k = config['CENTROIDS']
        # dir
        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                          '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'
        # self.test_path = self.root + 'rank/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
        #                   '/tgt_' + self.tgt + '_src_' + self.src + '/test.csv'
        # metrics
        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,'tgt_hit': 10, 'tgt_ndcg': 10,
                        'aug_mae': 10, 'aug_rmse': 10,'aug_hit': 10, 'aug_ndcg': 10,
                        'emcdr_mae': 10, 'emcdr_rmse': 10,'emcdr_hit': 10, 'emcdr_ndcg': 10,
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10,'ptupcdr_hit': 10, 'ptupcdr_ndcg': 10,
                        'metarec_mae':10, 'metarec_rmse':10,'metarec_hit': 10, 'metarec_ndcg': 10
                        }

    def get_model(self):
        """base pretrain"""
        if self.base_model == 'MF':
            model = MFBasedModelTwo(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'DNN':
            model = DNNBasedModelTwo(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'GMF':
            model = GMFBasedModelTwo(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def seq_extractor(self, x):
        """seq to array"""
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)




    def read_log_data(self, path, batchsize, history=False, vis=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else: # iter中多了positive sequence
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq', 'neg_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            if vis:
                data = data.drop_duplicates(subset=['uid'])
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=config['his'],
                                                                 padding='post') # B,20
            neg_seq = keras.preprocessing.sequence.pad_sequences(data.neg_seq.map(self.seq_extractor), maxlen=config['his'],
                                                                 padding='post')  # B,20
            # pos_seq = pad_sequence(datas,batch_first=True,padding_value=0)
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            neg_seq = torch.tensor(neg_seq, dtype=torch.long)

            id_fea = torch.tensor(data[x_col].values, dtype=torch.long) # 0,1 位是uid, iid
            X = torch.cat([id_fea, pos_seq, neg_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            """if path.split('/')[-1]=="test.csv":
                data_iter = DataLoader(dataset, batchsize, shuffle=False, drop_last=True) #for rank
            else:
                data_iter = DataLoader(dataset, batchsize, shuffle=True))"""
            data_iter = DataLoader(dataset, batchsize, shuffle=True
            return data_iter

    def split_list(self, row):
        index = len(row['items'])//2  # 1
        return row['items'][:index], row['items'][index:] # 只使用一半进行finetune

    def read_warm(self, path, batchsize):
        """ warm and finetune
        """
        data = pd.read_csv(path, header=None)
        cols = ['uid', 'iid', 'y', 'pos_seq', 'neg_seq']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data.columns = cols
        uid_set = set(data.uid.unique())
        items_seq_dict, y_seq_dict = get_history(data, uid_set)
        data['items'] = data['uid'].map(items_seq_dict)
        data[['finetune_items', 'test_item']] = data.apply(self.split_list, axis=1, result_type='expand')

        # mask = (data['iid'] == data['finetune_items'])
        mask = data.apply(lambda x: x['iid'] in x['finetune_items'], axis=1)
        test_data = data[~mask].copy()
        finetune_data = data[mask].copy()

        pos_seq = keras.preprocessing.sequence.pad_sequences(test_data.pos_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        neg_seq = keras.preprocessing.sequence.pad_sequences(test_data.neg_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        pos_seq = torch.tensor(pos_seq, dtype=torch.long)
        neg_seq = torch.tensor(neg_seq, dtype=torch.long)
        id_fea = torch.tensor(test_data[x_col].values, dtype=torch.long)  # 0,1 =>uid, iid
        X = torch.cat([id_fea, pos_seq, neg_seq], dim=1)
        y = torch.tensor(test_data[y_col].values, dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)

        # 用于finetune
        pos_seq = keras.preprocessing.sequence.pad_sequences(finetune_data.pos_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        neg_seq = keras.preprocessing.sequence.pad_sequences(finetune_data.neg_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        pos_seq = torch.tensor(pos_seq, dtype=torch.long)
        neg_seq = torch.tensor(neg_seq, dtype=torch.long)
        id_fea = torch.tensor(finetune_data[x_col].values, dtype=torch.long)  
        X = torch.cat([id_fea, pos_seq, neg_seq], dim=1)
        y = torch.tensor(finetune_data[y_col].values, dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset_finetune = TensorDataset(X, y)
        data_finetune_history = DataLoader(dataset_finetune, batchsize, shuffle=True) 

        X = id_fea
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset_finetune = TensorDataset(X, y)
        data_finetune_nohistory = DataLoader(dataset_finetune, batchsize, shuffle=True)
        return data_iter, data_finetune_nohistory, data_finetune_history


    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq', 'neg_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        """data aug base"""
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_aug = self.read_aug_data()
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        # several matrix
        others_data = self.other_data(self.src_path, self.meta_path, self.test_path)

        return data_src, data_tgt, data_meta, data_map, data_aug, data_test, others_data # data_meta和data_test都有sample 版本

    def other_data(self, src, meta, test):
        """use for sampling aug, torch.long"""
        cols = ['uid', 'iid', 'y']
        data_src = pd.read_csv(src, header=None)
        data_src.columns = cols
        cols = ['uid', 'iid', 'y', 'pos_seq', 'neg_seq']
        data_test = pd.read_csv(test, header=None)
        data_test.columns = cols
        data_meta = pd.read_csv(meta, header=None)
        data_meta.columns = cols
        data_all = pd.concat([data_src[['uid','iid']], data_test[['uid','iid']]], axis=0)
        popularity_items = data_all.groupby('iid')['uid'].count().reset_index().sort_values(by='uid',ascending=False)['iid'].values
        popularity_probs = data_all.groupby('iid')['uid'].count().reset_index().sort_values(by='uid',ascending=False)['uid'].values
        popularity_items = torch.tensor(popularity_items[:config['POP']], dtype=torch.long)
        popularity_probs = torch.tensor(popularity_probs[:config['POP']], dtype=torch.float32)

        co_users = torch.tensor(np.union1d(data_meta['uid'].unique(), data_test['uid'].unique()), dtype=torch.long)
        if self.use_cuda:
            co_users = co_users.cuda()
            popularity_items = popularity_items.cuda()
            popularity_probs = popularity_probs.cuda()
        return co_users, popularity_items, popularity_probs


    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd) # /10 rank
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd) # /10
        optimizer_meta = torch.optim.Adam(params=list(model.meta_net.parameters())+list(model.general_bridge.parameters()), lr=self.lr, weight_decay=self.wd) 
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.emb_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze()
        if self.use_cuda:
            centroids, node2cluster = centroids.cuda(), node2cluster.cuda()

        return centroids, node2cluster


    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False, aug_samples=None, prototype=None):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            elif stage == 'train_meta':
                pred, triple_loss, proto_loss,_,_ = model((X, aug_samples, prototype), stage)
                loss = criterion(pred, y.squeeze().float()) 
                loss += triple_loss * config['triple_reg']
                loss += proto_loss * config['proto_reg']
#                 print(loss,  proto_loss * config['proto_reg'])
            else:
                pred, emb_loss = model(X, stage)
                loss = criterion(pred, y.squeeze().float())
                loss += emb_loss


            model.zero_grad()
            loss.backward()
            optimizer.step()

    def eval_mae(self, model, data_loader, stage, aug_samples=None, prototype=None):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                if stage == 'test_meta':
                    pred = model((X, aug_samples, prototype), stage)
                    pred = pred[0]
                else:
                    pred,_ = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item() # MSE, RMSE



    def eval_mae_p(self, model, data_loader, stage):
        """use for ptupcdr"""
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred,_,_ = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item() # MSE, RMSE

    def train_p(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
        """use for ptupcdr"""
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred,emb_loss,_ = model(X, stage)
                loss = criterion(pred, y.squeeze().float())
                loss += emb_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer, warm_data=False):
        print('=========TgtOnly========')
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            # if warm_data:
            #     self.train(warm_data, model, criterion, optimizer, i, stage='train_tgt')
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            print('MAE: {} RMSE: {}'.format(mae, rmse))


    def CDR(self, model, data_src, data_map, data_meta, data_test, others_data,
            criterion, optimizer_src, optimizer_map, optimizer_meta, warm_data=False):
        if not warm_data:
            print('=====CDR Pretraining=====')
            for i in range(self.epoch):
                self.train(data_src, model, criterion, optimizer_src, i, stage='train_src') 
        print('=====Source Centroids=====')
        if config['BASE_MODEL'] != 'MF':
            centroids, node2cluster = self.run_kmeans(model.src_model.embedding.uid_embedding.weight.data.cpu().numpy())
        else:
            centroids, node2cluster = self.run_kmeans(model.src_model.uid_embedding.weight.data.cpu().numpy())

        print('=====Sample Augmentation=====') 
        ranklists_top, ranklists_tail, ranklists_top_prob, ranklists_tail_prob = self.eval_prob(model, others_data, stage='test_prob') # [Co-users, sample_all]

        print('==========MetaRec==========')
        for i in range(self.epoch):
            aug_samples = self.sampling_aug((ranklists_top_prob, ranklists_tail_prob), (ranklists_top, ranklists_tail), others_data[0])
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta', aug_samples=aug_samples, prototype=(centroids, node2cluster))
            # if warm_data:
            #     self.train(warm_data, model, criterion, optimizer_meta, i, stage='train_meta', aug_samples=aug_samples,
            #                prototype=(centroids, node2cluster))
            mae, rmse = self.eval_mae(model, data_test, stage='test_meta', aug_samples=aug_samples, prototype=(centroids, node2cluster))
            self.update_results(mae, rmse, 'metarec')
            print('MAE: {} RMSE: {}'.format(mae, rmse))
            
        data_vis = self.read_log_data(self.meta_path, 1000, history=True,vis=True)
        tgt_emb_d,trans_emb_d,src_emb_d = np.empty((0,10)), np.empty((0,10)),np.empty((0,10))
        for X,y in data_vis:
            aug_samples = self.sampling_aug((ranklists_top_prob, ranklists_tail_prob), (ranklists_top, ranklists_tail), others_data[0])
            src_emb = model.src_model.uid_embedding(X[:, 0])
            tgt_emb = model.tgt_model.uid_embedding(X[:, 0])
            src_emb_d= np.vstack((src_emb_d,src_emb.cpu().detach().numpy()))
            tgt_emb_d= np.vstack((tgt_emb_d,tgt_emb.cpu().detach().numpy()))
            emb = model((X, aug_samples, (centroids, node2cluster)), 'test_meta')
            trans_emb = emb[-1][:, 0, :]
            trans_emb_d = np.vstack((trans_emb_d,trans_emb.cpu().detach().numpy()))
        pd.DataFrame(src_emb_d).to_csv('../draw/src_cvpm_enhance2.csv')
        pd.DataFrame(tgt_emb_d).to_csv('../draw/tgt_cvpm_enhance2.csv')
        pd.DataFrame(trans_emb_d).to_csv('../draw/trans_cvpm_enhance2.csv')


    def eval_prob(self, model, data, stage):
        """获取source域user对于所有采样的item的偏好程度"""
        # candidate pool
        model.eval()
        with torch.no_grad():
            scores, data, popularity = model(data, stage)
            prob_matrix = np.argsort(-scores)
            # ranklists_top_prob, ranklists_tail_prob = prob_matrix[:, :config['SAMPLE']], prob_matrix[:,
            #                                                                             -config['SAMPLE']:] # D, sample
            ranklists_top = torch.tensor(np.take_along_axis(data, prob_matrix, axis=-1)[:, :config['SAMPLE']], dtype=torch.float32)
            ranklists_tail = torch.tensor(np.take_along_axis(data, prob_matrix, axis=-1)[:, -config['SAMPLE']:], dtype=torch.float32) # D, sample

            # popularity
            if config['sample_strategy']=="pop":
                ranklists_top_prob, ranklists_tail_prob = ranklists_top, ranklists_tail
                # ranklists_top_prob = torch.tensor(np.take_along_axis(popularity, prob_matrix, axis=-1)[:, :config['SAMPLE']],
                #                              dtype=torch.float32) + 1e-6
                # ranklists_tail_prob = torch.tensor(np.take_along_axis(popularity, prob_matrix, axis=-1)[:, -config['SAMPLE']:],
                #                               dtype=torch.float32) +1e-6 # D, sample

            elif config['sample_strategy']=="tf-idf":
                # ranklists_top_prob, ranklists_tail_prob = ranklists_top, ranklists_tail
                ranklists_top_prob = 1/(torch.tensor(np.take_along_axis(popularity, prob_matrix, axis=-1)[:, :config['SAMPLE']],
                                             dtype=torch.float32) + 1e-6)
                ranklists_tail_prob = torch.tensor(np.take_along_axis(popularity, prob_matrix, axis=-1)[:, -config['SAMPLE']:],
                                              dtype=torch.float32) +1e-6
            elif config['sample_strategy']=="random":
                ranklists_top_prob = torch.rand_like(ranklists_top)
                ranklists_tail_prob = torch.rand_like(ranklists_tail)
            if config['USE_CUDA']:
                ranklists_top, ranklists_tail = ranklists_top.cuda(), ranklists_tail.cuda()
                ranklists_top_prob, ranklists_tail_prob = ranklists_top_prob.cuda(), ranklists_tail_prob.cuda()
        return ranklists_top, ranklists_tail, ranklists_top_prob, ranklists_tail_prob

    def sampling_aug(self, weights, items_lists, co_users):
        """用于扩充positive和negative sequence"""
        # 根据概率采样
        top_indices = torch.multinomial(weights[0], config['TOPK']) # Co-user, K
        tail_indices = torch.multinomial(weights[1], config['TOPK']) # Co-user, K
        top_k_supplement = torch.gather(items_lists[0], 1, top_indices)
        tail_k_supplement = torch.gather(items_lists[1], 1, tail_indices)
        top_k = torch.nn.Embedding(self.uid_all, config['TOPK']) # detach()
        tail_k = torch.nn.Embedding(self.uid_all, config['TOPK'])
        torch.nn.init.zeros_(top_k.weight.data)
        torch.nn.init.zeros_(tail_k.weight.data)
        if config['USE_CUDA']:
            top_k, tail_k = top_k.cuda(), tail_k.cuda()
        top_k.weight.data[co_users] = top_k_supplement
        tail_k.weight.data[co_users] = tail_k_supplement
        return top_k, tail_k


    def main(self):
        model = self.get_model()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test, others_data = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)
        criterion = torch.nn.MSELoss()

        data_history, data_nohistory = [], []
        if config['WARM']:
            print("==============Finetune Process Start===================")
            data_test, data_nohistory, data_history = self.read_warm(self.test_path, self.batchsize_test)
            print("==============Finetune Process End===================")


        self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt, warm_data=data_nohistory) # pretrain
        self.CDR(model, data_src, data_map, data_meta, data_test, others_data,
                 criterion, optimizer_src, optimizer_map, optimizer_meta) # pretrain+ meta
        

         



# ======================= rank 
    # read_log_data要修改dataloader的drop, ready->rank, config

    def read_warm_rank(self, test_path, finetune_path, batchsize):
        test_data = pd.read_csv(finetune_path, header=None)
        finetune_data = pd.read_csv(test_path, header=None)
        cols = ['uid', 'iid', 'y', 'pos_seq', 'neg_seq']
        x_col = ['uid', 'iid']
        y_col = ['y']
        test_data.columns = cols
        finetune_data.columns = cols

        pos_seq = keras.preprocessing.sequence.pad_sequences(test_data.pos_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        neg_seq = keras.preprocessing.sequence.pad_sequences(test_data.neg_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        pos_seq = torch.tensor(pos_seq, dtype=torch.long)
        neg_seq = torch.tensor(neg_seq, dtype=torch.long)
        id_fea = torch.tensor(test_data[x_col].values, dtype=torch.long)  
        X = torch.cat([id_fea, pos_seq, neg_seq], dim=1)
        y = torch.tensor(test_data[y_col].values, dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, 10000, shuffle=False, drop_last=True) # 100 X test

        # for finetune
        pos_seq = keras.preprocessing.sequence.pad_sequences(finetune_data.pos_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        neg_seq = keras.preprocessing.sequence.pad_sequences(finetune_data.neg_seq.map(self.seq_extractor), maxlen=config['his'],
                                                             padding='post')  # B,20
        pos_seq = torch.tensor(pos_seq, dtype=torch.long)
        neg_seq = torch.tensor(neg_seq, dtype=torch.long)
        id_fea = torch.tensor(finetune_data[x_col].values, dtype=torch.long)  
        X = torch.cat([id_fea, pos_seq, neg_seq], dim=1)
        y = torch.tensor(finetune_data[y_col].values, dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset_finetune = TensorDataset(X, y)
        data_finetune_history = DataLoader(dataset_finetune, batchsize, shuffle=True)  # batchsize为1
        X = id_fea
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset_finetune = TensorDataset(X, y)
        data_finetune_nohistory = DataLoader(dataset_finetune, batchsize, shuffle=True)
        return data_iter, data_finetune_nohistory, data_finetune_history

    def CDR_rank(self, model, data_src, data_map, data_meta, data_test, others_data,
            criterion, optimizer_src, optimizer_map, optimizer_meta, warm_data=False):
        if not warm_data:
            print('=====CDR Pretraining=====')
            for i in range(self.epoch):
                self.train(data_src, model, criterion, optimizer_src, i, stage='train_src') 

        print('=====Source Centroids=====')
        if config['BASE_MODEL'] != 'MF':
            centroids, node2cluster = self.run_kmeans(model.src_model.embedding.uid_embedding.weight.data.cpu().numpy())
        else:
            centroids, node2cluster = self.run_kmeans(model.src_model.uid_embedding.weight.data.cpu().numpy())
            
        print('=====Sample Augmentation=====') 
        ranklists_top, ranklists_tail, ranklists_top_prob, ranklists_tail_prob = self.eval_prob(model, others_data, stage='test_prob') # [Co-users, sample_all]
        

        print('==========MetaRec==========')
        for i in range(self.epoch):
            aug_samples = self.sampling_aug((ranklists_top_prob, ranklists_tail_prob), (ranklists_top, ranklists_tail), others_data[0])
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta', aug_samples=aug_samples, prototype=(centroids, node2cluster))
#             if warm_data:
#                 self.train(warm_data, model, criterion, optimizer_meta, i, stage='train_meta', aug_samples=aug_samples,
#                            prototype=(centroids, node2cluster))
            hit, ndcg = self.eval_rank(model, data_test, stage='test_meta', aug_samples=aug_samples, prototype=(centroids, node2cluster))
            self.update_results(hit, ndcg, 'metarec')
            print('Hit: {} NDCG: {}'.format(hit, ndcg))
            
        data_vis = self.read_log_data(self.meta_path, 100, history=True,vis=True)
        tgt_emb_d,trans_emb_d,src_emb_d = np.empty((0,10)), np.empty((0,10)),np.empty((0,10))
        for X,y in data_vis:
            aug_samples = self.sampling_aug((ranklists_top_prob, ranklists_tail_prob), (ranklists_top, ranklists_tail), others_data[0])
            src_emb = model.src_model.uid_embedding(X[:, 0])
            tgt_emb = model.tgt_model.uid_embedding(X[:, 0])
            src_emb_d= np.vstack((src_emb_d,src_emb.cpu().detach().numpy()))
            tgt_emb_d= np.vstack((tgt_emb_d,tgt_emb.cpu().detach().numpy()))
            emb = model((X, aug_samples, (centroids, node2cluster)), 'test_meta')
            trans_emb = emb[-1][:, 0, :]
            trans_emb_d = np.vstack((trans_emb_d,trans_emb.cpu().detach().numpy()))
        pd.DataFrame(src_emb_d).to_csv('../draw/src_cvpm_rank_enhance2.csv')
        pd.DataFrame(tgt_emb_d).to_csv('../draw/tgt_cvpm_rank_enhance2.csv')
        pd.DataFrame(trans_emb_d).to_csv('../draw/trans_cvpm_rank_enhance2.csv')
        



    def eval_rank_p(self, model, data_loader, stage):
        print('Evaluating Rank:')
        model.eval()
        hit, ndcg = list(), list()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                pred = pred[0]

                batch_hit, batch_ndcg = calculate_metrics(pred.view(-1, 100), X[:,1].view(-1, 100))
                hit.extend(batch_hit)
                ndcg.extend(batch_ndcg)
        return np.mean(hit), np.mean(ndcg)  # Hit NDCG

    def eval_rank(self, model, data_loader, stage, aug_samples=None, prototype=None):
        print('Evaluating Hit and NDCG:')
        model.eval()
        hit, ndcg = list(), list()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                if stage == "test_meta":
                    scores = model((X, aug_samples, prototype), stage)
                    scores = scores[0]
                else:
                    scores = model(X, stage)
                    scores = scores[0]
                batch_hit, batch_ndcg = calculate_metrics(scores.view(-1,100), X[:,1].view(-1,100))
                hit.extend(batch_hit)
                ndcg.extend(batch_ndcg)
        return np.mean(hit), np.mean(ndcg)


    def TgtOnly_rank(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            hit, ndcg = self.eval_rank(model, data_test, stage='test_tgt')
            self.update_results_rank(hit, ndcg, 'tgt')
            print('Hit: {} NDCG: {}'.format(hit, ndcg))



    def update_results_rank(self, hit, ndcg, phase):
        if hit < self.results[phase + '_hit']:
            self.results[phase + '_hit'] = hit
        if ndcg < self.results[phase + '_ndcg']:
            self.results[phase + '_ndcg'] = ndcg


    def main_rank(self):
        self.input_root = self.root + 'rank/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                          '/tgt_' + self.tgt + '_src_' + self.src
        self.finetune_path = self.input_root + '/finetune.csv'
        model = self.get_model()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test, others_data = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map = self.get_optimizer(model)
        criterion = torch.nn.BCEWithLogitsLoss() # 
        # criterion = torch.nn.MSELoss() 

        data_history, data_nohistory = [], []
        if config['WARM']:
            print("==============Finetune Process Start===================")
            data_test, data_nohistory, data_history = self.read_warm_rank(self.test_path,self.finetune_path, self.batchsize_test)
            print("==============Finetune Process End===================")
        
            print("====================Concate Dataloder================")
            dataset_src = data_src.dataset
            dataset_tgt = data_tgt.dataset
            dataset_meta = data_meta.dataset
            dataset_nohistory = data_nohistory.dataset
            dataset_history = data_history.dataset
            concat_src = ConcatDataset([dataset_src, dataset_nohistory])
            concat_tgt = ConcatDataset([dataset_tgt, dataset_nohistory])
            concat_meta = ConcatDataset([dataset_meta, dataset_history])
            data_src = DataLoader(concat_src, batch_size=self.batchsize_src, shuffle=True)
            data_tgt = DataLoader(concat_tgt, batch_size=self.batchsize_tgt, shuffle=True)
            data_meta = DataLoader(concat_meta, batch_size=self.batchsize_meta, shuffle=True) 
            print("====================Concate ENd================")

        self.TgtOnly_rank(model, data_tgt, data_test, criterion, optimizer_tgt)  # target
        self.CDR_rank(model, data_src, data_map, data_meta, data_test, others_data,
                 criterion, optimizer_src, optimizer_map, optimizer_meta)  # source_meta



