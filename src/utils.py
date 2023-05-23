#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 10:51
# @Author  : Jack Zhao
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

# #Desc: several tools
import math
import random
import tqdm
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
from collections import defaultdict
from config import config
from torch.utils.data import Dataset
from datetime import datetime



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def tf_idf(u_i_matrix):
    pass


class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss * config['emb_reg']

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # anchor-positive
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        # anchor-neg
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        # loss
        loss = torch.mean(torch.nn.functional.relu(pos_dist - neg_dist + self.margin))
        return loss


def get_history(data, uid_set):
    """get positive {user:[item, item]}"""
    item_seq_dict = {}
    y_seq_dict = {}
    for uid in tqdm.tqdm(uid_set):
        items = data[(data.uid == uid)].iid.values.tolist()
        y = data[(data.uid == uid)].y.values.tolist()
        item_seq_dict[uid] = items
        y_seq_dict[uid] = y
    return item_seq_dict, y_seq_dict


def get_pos_neg(data, uid_set):
    """get positive {user:[item, item]}"""
    pos_seq_dict = {}
    neg_seq_dict = {}
    for uid in tqdm.tqdm(uid_set):
        pos = data[(data.uid == uid) & (data.y > 0)].iid.values.tolist()
        neg = data[(data.uid == uid) & (data.y <= 0)].iid.values.tolist()
        pos_seq_dict[uid] = pos
        neg_seq_dict[uid] = neg
    return pos_seq_dict, neg_seq_dict


def time2stamp(cmnttime):  #
    cmnttime = datetime.strptime(cmnttime, '%Y-%m-%d')
    stamp = int(datetime.timestamp(cmnttime))
    return stamp

class BPRDataset(Dataset):
    """baseline用"""
    def __init__(self, data, num_users, num_items):
        super().__init__()
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.u_dict = self.data.groupby('uid')['iid'].apply(set).to_dict()

    def __getitem__(self, index):
        row = self.data.iloc[index]
        user_id, item_id = row['uid'], row['iid']
        negative_item_id = np.random.randint(self.num_items)
        while negative_item_id in self.u_dict[user_id]:
            negative_item_id = np.random.randint(self.num_items)
        rows = torch.cat((torch.LongTensor([user_id]), torch.LongTensor([item_id]), torch.LongTensor([negative_item_id])), dim=-1)
        return (rows,torch.Tensor([row['y']])) if not config['USE_CUDA'] else (rows.cuda(),torch.Tensor([row['y']]).cuda())

    def __len__(self):
        return len(self.data)






##############下面是rank-based的一些数据处理方案

def train_negative_sampling(df, k=5):
    """data_src,data_tgt负采样"""
    users = df['uid'].unique()
    items = df['iid'].unique()

    # pos dict
    user_items = {}
    for user in users:
        user_items[user] = df[df['uid'] == user]['iid'].values

    samples = []
    for _, row in df.iterrows():
        user, pos_item, y = row['uid'], row['iid'], row['y']
        for _ in range(k):
            neg_item = np.random.choice(items)
            while neg_item in user_items[user]:
                neg_item = np.random.choice(items)
            samples.append((user, neg_item, 0))
        samples.append((user, pos_item, 1)) 

    new_df = pd.DataFrame(samples, columns=['uid', 'iid', 'y'])

    return new_df


def src_tgt_process(root):
    for root, dirs, files in os.walk(root):
        if root.split('\\')[-1]!="tgt_netflix_src_movie_lens":
            continue
        else:
            for file in files:
                read_files = os.path.join(root, file)
                typ = read_files.split('\\')[-1]
                if typ == 'train_src.csv' or typ == 'train_tgt.csv':
                    print(typ, read_files)
                    df = pd.read_csv(read_files, header=None)
                    df.columns = ['uid', 'iid', 'y']
                    df = train_negative_sampling(df, k=config['NEG_K'])
                    df[['uid', 'iid']] = df[['uid', 'iid']].astype('int')
                    file_save(read_files, df)
                elif typ == 'train_meta' or typ == 'test.csv':
                    continue

def file_save(read_files, df):
    out_files = sub_path(read_files, "rank", -4)
    dire, name = os.path.split(out_files)
    if not os.path.exists(dire):
        os.makedirs(dire)
    df.to_csv(out_files, sep=',', header=None, index=False)


def sub_path(path, sub, pos):
    """替换path"""
    out_files = path.split('\\')
    out_files[pos] = sub
    out_files = '\\'.join(out_files)
    return out_files

def split_list(row):
    index = -1  # 1 
    # index = len(row['interacted'])//2
    return [row['interacted'][-1]], row['interacted'][:index]


def test_negative_sampling(df, k, iid_range):
    """Test Negative sampling method"""
    users = df['uid'].unique()
    items = list(range(int(iid_range[0]),int(iid_range[1])))
    user_items = {}
    for user in users:
        user_items[user] = df[df['uid'] == user]['iid'].values

    df['interacted'] = df['uid'].map(user_items)
    df[['test_item', 'finetune_items']] = df.apply(split_list, axis=1, result_type='expand')

    mask = df.apply(lambda x: x['iid'] in x['finetune_items'], axis=1)
    test_df = df[~mask].copy()
    finetune_df = df[mask].copy()
    finetune_df = finetune_df[['uid', 'iid', 'y', 'pos_seq', 'neg_seq']]

    test_df[['pos_seq','neg_seq']] = test_df.apply(cut_len, axis=1,  result_type='expand')  
    finetune_df[['pos_seq','neg_seq']] = finetune_df.apply(cut_len, axis=1,  result_type='expand')  
    print(test_df.shape,finetune_df.shape)

    samples = []
    
    for _, row in test_df.iterrows():
        user, pos_item, y, pos_seq, neg_seq = row['uid'], row['iid'], row['y'], row['pos_seq'], row['neg_seq']
        samples.append((user, pos_item, y,  pos_seq, neg_seq))
        for _ in range(k):
            neg_item = np.random.choice(items)
            while neg_item in user_items[user]:
                neg_item = np.random.choice(items)
            samples.append((user, neg_item, 0, pos_seq, neg_seq))

    test_df = pd.DataFrame(samples, columns=['uid', 'iid', 'y', 'pos_seq', 'neg_seq'])

    return finetune_df, test_df

def cut_len(row):
    # pos_length = len(row['pos_seq'])
    return row['pos_seq'][:100], row['neg_seq'][:100] 


def meta_test_process(root):
    out_files = sub_path(root, 'rank', -1)
    min_subdirs = []
    for dirpath, dirnames, filenames in os.walk(out_files):
        if not dirnames:  
            min_subdirs.append(dirpath)
    print("min_subdirs",min_subdirs)

    # for dirpath in min_subdirs:
    #     print(dirpath)
    #     src = pd.read_csv(dirpath + '/train_src.csv', header=None)
    #     tgt = pd.read_csv(dirpath + '/train_tgt.csv', header=None)
    #     src.columns, tgt.columns = ['uid', 'iid', 'y'], ['uid', 'iid', 'y']
    #     src[['uid', 'iid']] = src[['uid', 'iid']].astype("int")
    #     tgt[['uid', 'iid']] = tgt[['uid', 'iid']].astype("int")
    #     src.to_csv(dirpath + '/train_src.csv', sep=',', header=None, index=False)
    #     tgt.to_csv(dirpath + '/train_tgt.csv', sep=',', header=None, index=False)
    # print('Transfer Done !')

    for dirpath in min_subdirs:
        if dirpath.split('\\')[-1] != "tgt_netflix_src_movie_lens": # selective process
            continue
        else:
            src = pd.read_csv(dirpath + '/train_src.csv', header=None)
            tgt = pd.read_csv(dirpath + '/train_tgt.csv', header=None)
            src.columns, tgt.columns = ['uid', 'iid', 'y'], ['uid', 'iid', 'y']
            origin_path = sub_path(dirpath, "ready", -3)
            origin_train_meta = pd.read_csv(origin_path + '/train_meta.csv', header=None)
            origin_train_meta.columns = ['uid', 'iid', 'y', 'pos', 'neg']
            co_users = origin_train_meta['uid'].unique()
            pos_seq_dict, neg_seq_dict = get_pos_neg(src, co_users) 
            train_meta = tgt[tgt['uid'].isin(co_users)]
            train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
            train_meta['neg_seq'] = train_meta['uid'].map(neg_seq_dict)
            train_meta[['pos_seq','neg_seq']] = train_meta.apply(cut_len, axis=1, result_type="expand") 
            train_meta.to_csv(dirpath + '/train_meta.csv', sep=',', header=None, index=False)


            origin_test = pd.read_csv(origin_path + '/test.csv', header=None)
            origin_test.columns = ['uid', 'iid', 'y', 'pos_seq', 'neg_seq']
            co_users = origin_test['uid'].unique()
            pos_seq_dict, neg_seq_dict = get_pos_neg(src, co_users) 
            origin_test['pos_seq'] = origin_test['uid'].map(pos_seq_dict) 
            origin_test['neg_seq'] = origin_test['uid'].map(neg_seq_dict)
            iid_min, iid_max = min(tgt['iid'].min(), origin_test['iid'].min()), max(tgt['iid'].max(), origin_test['iid'].max())
            finetune_df, test_df = test_negative_sampling(origin_test, config['EVAL_K'], [iid_min,iid_max])
            # finetune_df[['uid','iid']], test_df[['uid','iid']] = finetune_df[['uid','iid']].astype('int'), test_df[['uid','iid']].astype('int')
            finetune_df.to_csv(dirpath + '/finetune.csv', sep=',', header=None, index=False)
            test_df.to_csv(dirpath + '/test.csv', sep=',', header=None, index=False)


def getHIT(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0


def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i + 2)
    return 0


def reg(tensor):
    return torch.sqrt(torch.sum(torch.square(tensor), axis=1) + 1e-8)

def calculate_metrics(scores, batched_data):
    """计算指标,B,100,1"""
    funcs = {'hit':getHIT, 'ndcg': getNDCG}
    batched_data = batched_data.cpu().numpy()
    targets = batched_data[:, 0]
    scores = scores.cpu().numpy()
    result = defaultdict(list)
    for metric in ['hit', 'ndcg']:
        func = funcs[metric]
        ranklists = np.take_along_axis(batched_data, np.argsort(-scores), axis=-1)[:, :config['RANK_TOPK']]
        for target, ranklist in zip(targets, ranklists):
            tmp = func(ranklist, target)
            result[metric].append(tmp)
    return result['hit'], result['ndcg']



def bpr_dataset(pos_df, neg_df):
    pos_df = pos_df.loc[pos_df.index.repeat(config['NEG_K'])].reset_index(drop=True)
    neg_df['pos'] = pos_df['iid'].values
    return neg_df

if __name__ == '__main__':
    pass
    root = r'E:\Code\Pycharm\MapRec\data\ready'
    src_tgt_process(root)
    meta_test_process(root)











