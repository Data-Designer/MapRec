#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 10:51
# @Author  : Jack Zhao
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc: several config parameters


class MetaConfig():
    # data_info parameter
    src_tgt_pairs = {
        "1":
            {"src": "Movies_and_TV",
             "tgt": "CDs_and_Vinyl",
             "uid": 181187,
             "iid": 114495,
             "batchsize_src": 256,
             "batchsize_tgt": 256,
             "batchsize_meta": 128,
             "batchsize_map": 64,
             "batchsize_test": 128 
             },
        "2":
            {"src": "Books",
             "tgt": "Movies_and_TV",
             "uid": 690240,
             "iid": 418034,
             "batchsize_src": 512,
             "batchsize_tgt": 512,
             "batchsize_meta": 512,
             "batchsize_map": 128,
             "batchsize_test": 256
             },
        "3":
            {"src": "Books",
             "tgt": "CDs_and_Vinyl",
             "uid": 662188,
             "iid": 432425,
             "batchsize_src": 512,
             "batchsize_tgt": 512,
             "batchsize_meta": 512,
             "batchsize_map": 128,
             "batchsize_test": 256
             },
        "4":
            {"src": "movie",
             "tgt": "book",
             "uid": 2715,
             "iid": 130765,
             "batchsize_src": 256,
             "batchsize_tgt": 256,
             "batchsize_meta": 256,
             "batchsize_map": 64,
             "batchsize_test": 128
             },
        "5":
            {"src": "movie",
             "tgt": "music",
             "uid": 2717, 
             "iid": 114771, 
             "batchsize_src": 256,
             "batchsize_tgt": 256,
             "batchsize_meta": 256,
             "batchsize_map": 64,
             "batchsize_test": 128
             }, # douban dataset
        "6":
            {"src": "movie_lens",
             "tgt": "netflix",
             "uid": 14129,  
             "iid": 73584,  
             "batchsize_src": 512,
             "batchsize_tgt": 512,
             "batchsize_meta": 512,
             "batchsize_map": 128,
             "batchsize_test": 256
             },  # netflix
    }
    MID_DONE = True
    READY_DONE = True
    WARM = False
    RANK = True

    # train parameter
    TASK = '1' 
    BASE_MODEL = 'MF' 
    SEED = 2020
    RATIO = [0.8,0.2] 
    GPU = '0'
    USE_CUDA = True
    EPOCH = 8
    LR = 0.01 # task 1-3: 0.01 task6: 0.001 task4-5: 0.002/0.005;;;;;;;ptucdr task4:0.001
    EMB_DIM = 10
    META_DIM = 50
    NUM_FIELDS = 2
    WD = 0 
    POP = 1000 # informative item
    SAMPLE = 100 
    TOPK = 5 # select 5
    CENTROIDS = 100 # 100
    NEG_K = 4 
    EVAL_K = 99 # neg sample 99
    RANK_TOPK = 10 # ranklist

    # loss
    his = 20 
    emb_reg = 0 
    margin = 1e-3
    triple_reg = 1e-3 # task1-3 1e-3; task4 1
    proto_reg= 1e-3 
    ssl_temp = 0.1
    sample_strategy = 'tf-idf'

    # dir parameter
    ROOT = '/dfs/data/MapRec/data/'

class MetaConfig_Rank():
    # data_info parameter
    src_tgt_pairs = {
        "1":
            {"src": "Movies_and_TV",
             "tgt": "CDs_and_Vinyl",
             "uid": 181187,
             "iid": 114495,
             "batchsize_src": 2560,
             "batchsize_tgt": 2560,
             "batchsize_meta": 1280,
             "batchsize_map": 640,
             "batchsize_test": 12800 # 
             },
        "2":
            {"src": "Books",
             "tgt": "Movies_and_TV",
             "uid": 690240,
             "iid": 418034,
             "batchsize_src": 5120,
             "batchsize_tgt": 5120,
             "batchsize_meta": 5120,
             "batchsize_map": 1280,
             "batchsize_test": 25600
             },
        "3":
            {"src": "Books",
             "tgt": "CDs_and_Vinyl",
             "uid": 662188,
             "iid": 432425,
             "batchsize_src": 5120,
             "batchsize_tgt": 5120,
             "batchsize_meta": 5120,
             "batchsize_map": 1280,
             "batchsize_test": 25600
             },
        "4":
            {"src": "movie",
             "tgt": "book",
             "uid": 2715,
             "iid": 130765,
             "batchsize_src": 2560,
             "batchsize_tgt": 2560,
             "batchsize_meta": 2560,
             "batchsize_map": 640,
             "batchsize_test": 12800
             },
        "5":
            {"src": "movie",
             "tgt": "music",
             "uid": 2717, # max
             "iid": 114771, # sum
             "batchsize_src": 2560,
             "batchsize_tgt": 2560,
             "batchsize_meta": 2560,
             "batchsize_map": 640,
             "batchsize_test": 12800
             }, # douban dataset
        "6":
            {"src": "movie_lens",
             "tgt": "netflix",
             "uid": 14129,  
             "iid": 73584,  
             "batchsize_src": 5120,
             "batchsize_tgt": 5120,
             "batchsize_meta": 5120,
             "batchsize_map": 1280,
             "batchsize_test": 1000
             },  # netflix
    }
    MID_DONE = True
    READY_DONE = True
    WARM = False
    RANK = True

    # train parameter
    TASK = '1'
    BASE_MODEL = 'MF'
    SEED = 2020
    RATIO = [0.8, 0.2] 
    GPU = '0'
    USE_CUDA = True
    EPOCH = 10
    LR = 0.005 # task 1 0.005;2-3: 0.001 task4 0.005; 5 0.003 task 6 0.01
    EMB_DIM = 10
    META_DIM = 50
    NUM_FIELDS = 2
    WD = 0 
    POP = 1000 # informative item
    SAMPLE = 100 
    TOPK = 5 
    CENTROIDS = 100
    NEG_K = 4 
    EVAL_K = 99 
    RANK_TOPK = 10 


    # loss
    his = 20 
    emb_reg = 0.01 # task 1-3: 0.01 task 4 0.1
    margin = 1e-3
    triple_reg = 1e-2
    proto_reg= 1e-2
    ssl_temp = 0.1
    sample_strategy = 'tf-idf'

    # dir parameter
    ROOT = '/dfs/data/MapRec/data/'



# config = vars(MetaConfig_Rank)
config = vars(MetaConfig)

config = {k:v for k,v in config.items() if not k.startswith('__')}

