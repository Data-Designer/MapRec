#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 10:52
# @Author  : Jack Zhao
# @Site    : 
# @File    : dataload.py
# @Software: PyCharm

# #Desc: data-related preprocess
import pandas as pd
import numpy as np
import gzip
import json
import tqdm
import random
import os
from utils import time2stamp



class DataToMid():
    """unzip raw data to Dataframe"""
    def __init__(self, root, dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print('Parsing '+ self.dealing + ' to mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                re.append([line['reviewerID'], line['asin'], line['overall']])
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y']) # u,i,score
        print(self.dealing + ' Dataframe Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re

    def douban_main(self):
        """douban"""
        print('Parsing ' + self.dealing + ' to mid...')
        re = pd.read_table(self.root + 'raw/douban/' + self.dealing + 'reviews_cleaned' + '.txt')
        # print(re.head(5))
        select_cols = re.columns[0:3]
        re = re[select_cols]
        re.columns = ['uid', 'iid', 'y']
        print(self.dealing + ' Dataframe Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re

    def netflix_pro(self):
        """netflix"""
        df1 = pd.read_csv(self.root + 'raw/' + self.dealing +'/combined_data_1.txt', header=None, names=['user_id', 'rating', 'timestamp'],
                          usecols=[0, 1, 2])
        df2 = pd.read_csv(self.root + 'raw/' + self.dealing + '/combined_data_2.txt', header=None,
                          names=['user_id', 'rating', 'timestamp'],
                          usecols=[0, 1, 2])
        df3 = pd.read_csv(self.root + 'raw/' + self.dealing + '/combined_data_3.txt', header=None,
                          names=['user_id', 'rating', 'timestamp'],
                          usecols=[0, 1, 2])
        df4 = pd.read_csv(self.root + 'raw/' + self.dealing + '/combined_data_4.txt', header=None,
                          names=['user_id', 'rating', 'timestamp'],
                          usecols=[0, 1, 2])
        df1['rating'] = df1['rating'].astype(float)
        df2['rating'] = df2['rating'].astype(float)
        df3['rating'] = df3['rating'].astype(float)
        df4['rating'] = df4['rating'].astype(float)
        df = df1
        df.append(df2)
        df.append(df3)
        df.append(df4)
        df.index = np.arange(0, len(df))
        print('Full dataset shape: {}'.format(df.shape))
        print('-Dataset examples-')
        df_nan = pd.DataFrame(pd.isnull(df.rating))
        df_nan = df_nan[df_nan['rating'] == True]
        df_nan = df_nan.reset_index()

        # data clean
        item_np = []
        item_id = 1
        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            
            temp = np.full((1, i - j - 1), item_id)
            item_np = np.append(item_np, temp)
            item_id += 1
        last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), item_id)
        item_np = np.append(item_np, last_record)

        df = df[pd.notnull(df['rating'])].copy()
        df['item_id'] = item_np.astype(int)
        df['user_id'] = df['user_id'].astype(int)
        df = df.loc[:, ['user_id', 'item_id', 'rating', 'timestamp']]  # swap
        df['timestamp'] = df['timestamp'].astype(str).apply(time2stamp)  
        print('-Dataset examples-')
        print(df.iloc[::5000000, :])
        df.to_csv(self.root + 'raw/' + self.dealing + './ratings.csv', sep=',', index=0, header=0)

    def cst_main(self):
        print('Parsing ' + self.dealing + ' to mid...')
        if self.dealing == "movie_lens":
            re = pd.read_csv(self.root + 'raw/' + self.dealing + '/rating' + '.csv')
            select_cols = re.columns[0:3]
            re = re[select_cols]
            re.columns = ['uid', 'iid', 'y']
            print("Before filtering {}".format(re.shape))
            re = re.groupby('uid').filter(lambda x: len(x) >= 150).groupby('iid').filter(lambda x: len(x) >= 30)
            re = re.groupby('iid').filter(lambda x: len(x) <= 5000)
            print("After filtering {}".format(re.shape))
            print(self.dealing + ' Dataframe Done.')
            re.columns = ['iid','uid','y']

        elif self.dealing == "netflix":
            if not os.path.exists(self.root + 'raw/' + self.dealing + '/ratings' + '.csv'):
                self.netflix_pro()
            print("Netflix has been Done! ")
            # movie——lens
            movie_id = pd.read_csv(self.root + 'raw/movie_lens' + '/movie' + '.csv', header=0, index_col=None,delimiter=',')
            movie_id['title'] = movie_id['title'].str[:-7]

            dic = movie_id.set_index('title')['movieId'].to_dict()

            # netflix--name
            netflix_id = pd.read_csv(self.root + 'raw/'+self.dealing+"/movie_titles.csv", header = None, names = ['Movie_Id', 'Year', 'Name'],encoding = "ISO-8859-1",on_bad_lines='skip')
            netflix_id.columns = ['movieId','time','title']
            netflix_dic = netflix_id.set_index('movieId')['title'].to_dict()

            re = pd.read_csv(self.root + 'raw/' + self.dealing + '/ratings' + '.csv')
            select_cols = re.columns[0:3]
            re = re[select_cols]
            re.columns = ['uid', 'iid', 'y']
            # filter
            print("Before filtering {}".format(re.shape))
            re = re.groupby('uid').filter(lambda x: len(x) >= 150).groupby('iid').filter(lambda x: len(x) >= 15) 
            re = re.groupby('iid').filter(lambda x: len(x) <= 4000)
            re = re[re['iid'].isin(list(netflix_dic.keys()))]
            print("After filtering {}".format(re.shape))

            re['title'] = re['iid'].apply(lambda x: netflix_dic[x])

            counter = 131263 # movie_lens max
            def map_or_renumber(x):
                nonlocal counter
                if x in dic:
                    return dic[x]
                else:
                    dic[x] = counter
                    counter += 1
                    return dic[x]
            re['iid'] = re['title'].apply(map_or_renumber)
            print(re.head())
            re = re[re.columns[0:3]]
            re.columns = ['iid','uid','y'] # swap
            print(self.dealing + ' Dataframe Done.')

        re = re.reindex(columns=['uid', 'iid', 'y'])
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re


    def book_main(self):
        """cross system"""
        print('Parsing ' + self.dealing + ' to mid...')
        re = pd.read_table(self.root + 'raw/book/' + self.dealing + '/' + self.dealing + '.inter')
        if self.dealing == 'Book-Crossing':
            re.columns = ['iid', 'uid', 'y'] 
            print(self.dealing + ' Dataframe Done.')
        elif self.dealing == 'Librarything':
            isn_data = pd.read_table(self.root + 'raw/book/' +'Book-Crossing/Book-Crossing.item')
            dic = isn_data[['title:token_seq','ISBN:token']].to_dict()
            re.columns = ['uid', 'name', 'y']
            print(self.dealing + ' Dataframe Done.')
            counter = 0
            def map_or_renumber(x):
                nonlocal counter
                if x in dic:
                    return dic[x]
                else:
                    dic[x] = counter
                    counter += 1
                    return dic[x]

            re['iid'] = re['name'].apply(map_or_renumber)
            re = re[['uid','iid','y']]
            re.columns = ['iid','uid','y']

        re = re.reindex(columns=['uid', 'iid', 'y'])
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re



class DataToReady():
    def __init__(self, root, src_tgt_pairs, task, ratio):
        self.root = root
        self.src, self.tgt = src_tgt_pairs[task]['src'], src_tgt_pairs[task]['tgt']
        self.ratio = ratio

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        """new id, tgt item from [src_item, all_item]"""
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt

    def get_history(self, data, uid_set):
        """get positive {user:[item, item]}"""
        pos_seq_dict = {}
        neg_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):
            pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            neg = data[(data.uid == uid) & (data.y <= 3)].iid.values.tolist()
            pos_seq_dict[uid] = pos
            neg_seq_dict[uid] = neg
        return pos_seq_dict, neg_seq_dict

    def split(self, src, tgt):
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
        co_users = src_users & tgt_users
        test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users)))) # user
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
        test = tgt[tgt['uid'].isin(test_users)] # dataframe, warm start 
        pos_seq_dict, neg_seq_dict = self.get_history(src, co_users) 
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
        train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
        train_meta['neg_seq'] = train_meta['uid'].map(neg_seq_dict)
        test['pos_seq'] = test['uid'].map(pos_seq_dict) # id index
        test['neg_seq'] = test['uid'].map(neg_seq_dict) 
        return train_src, train_tgt, train_meta, test 

    def save(self, train_src, train_tgt, train_meta, test):
        output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        train_meta.to_csv(output_root +  '/train_meta.csv', sep=',', header=None, index=False)
        test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)
        print(output_root)


    def main(self):
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
        src, tgt = self.mapper(src, tgt)
        train_src, train_tgt, train_meta, test = self.split(src, tgt)
        self.save(train_src, train_tgt, train_meta, test)

