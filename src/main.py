#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 10:51
# @Author  : Jack Zhao
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc: main run; 
import os
from utils import seed_everything, src_tgt_process, meta_test_process
from config import config
from trainer import MetaRecTrainer
from dataload import DataToMid, DataToReady

def run_single_model(config):
    # log initial
    print("Hello MetaRec")
    os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU']


    # dataset
    if not config['MID_DONE']:
        # for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
        #     DataToMid(config['ROOT'], dealing).main()
        # for dealing in ['book', 'movie', 'music']:
        #     DataToMid(config['ROOT'], dealing).douban_main()
        for dealing in ['netflix', 'movie_lens']:#
            DataToMid(config['ROOT'], dealing).cst_main()

    if not config['READY_DONE']:
        for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]: # train:test
            for task in ['6']: # ['1','2','3', '4', '5', '6']
                DataToReady(config['ROOT'], config['src_tgt_pairs'], task, ratio).main()
    print('task:{}; model:{}; ratio:{}; epoch:{}; lr:{}; gpu:{}; seed:{};'.
          format(config['TASK'], config['BASE_MODEL'], config['RATIO'], config['EPOCH'], config['LR'], config['GPU'], config['SEED']))

    if not config['RANK']:
        """用新的数据集"""
        root = r'E:\Code\Pycharm\MapRec\data\ready'
        src_tgt_process(root)
        meta_test_process(root)

    # model initial
    trainer = MetaRecTrainer(config)

    # model training & eval
    if config['MID_DONE'] and config['READY_DONE']:
        trainer.main() # mae
        # trainer.main_rank() 
        # trainer.main_bpr()

    # info
    print("Everything is OK! ")

if __name__ == '__main__':
    config = config
    seed_everything(config['SEED'])
    run_single_model(config)
