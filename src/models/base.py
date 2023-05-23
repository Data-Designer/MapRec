#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 15:08
# @Author  : Jack Zhao
# @Site    : 
# @File    : base.py
# @Software: PyCharm

# #Desc: base model
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import TripletLoss, reg, EmbLoss
from config import config

class GMFBase(torch.nn.Module):
    """GMF base"""
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)
        self.reg_loss = EmbLoss()


    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        emb_loss = self.reg_loss(emb)[0]
        x = self.linear(x)
        return x.squeeze(1), emb_loss


class DNNBase(torch.nn.Module):
    """DNN base"""
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.reg_loss = EmbLoss()


    def forward(self, x):
        emb = self.embedding.forward(x)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        emb_loss = self.reg_loss(emb)[0]
        return x, emb_loss



class LookupEmbedding(torch.nn.Module):
    """create embedding lookup table"""
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim) 

        nn.init.xavier_normal_(self.uid_embedding.weight)
        nn.init.xavier_normal_(self.iid_embedding.weight) 

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1)) 
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1)) # B, 1, H
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False)) # character encoder
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim)) # meta-network

    def forward(self, emb_fea, seq_index):
        """emb_fea: [B,T,H], seq_index: [B,T]"""
        mask = (seq_index == 0).float() # 0为padding index , B,T, True==1. False==0
        event_K = self.event_K(emb_fea) # B,T,1
        t = event_K - torch.unsqueeze(mask, 2) * 1e8 # B,T,1
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1) # B, 1, H
        output = self.decoder(his_fea) # B, 1, H^2
        return output.squeeze(1)



class MetaNetThree(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super(MetaNetThree, self).__init__()
        self.emb_dim = emb_dim
        self.event_pos = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))  # character encoder
        self.event_neg = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                             torch.nn.Linear(emb_dim, 1, False))  # character encoder
        self.event_softmax = torch.nn.Softmax(dim=1)

        self.W_s = nn.Linear(2*emb_dim, emb_dim,bias=False)
        self.W_mu = nn.Linear(emb_dim, emb_dim,bias=False)
        self.W_sigma = nn.Linear(emb_dim, emb_dim,bias=False)

        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim*emb_dim))  # meta-network

    def generateEpsilon(self):
        return torch.randn(size=(self.emb_dim, )).cuda() if config['USE_CUDA'] else torch.randn(size=(self.emb_dim, ))

    def forward(self, emb_fea, seq_index):
        """emb_fea: ([B,T,H],[B,T,H]), seq_index: ([B,T],[B,T])
        return
        """
        # encoder
        mask_pos = (seq_index[0] == 0).float() # 0为padding index , B,T, True==1. False==0
        event_pos = self.event_pos(emb_fea[0]) # B,T,1
        t_pos = event_pos - torch.unsqueeze(mask_pos, 2) * 1e8 # B,T,1
        att_pos = self.event_softmax(t_pos)
        his_fea_pos = torch.sum(att_pos * emb_fea[0], 1).squeeze(1) # B, H

        mask_neg = (seq_index[1] == 0).float() # 0为padding index , B,T, True==1. False==0
        event_neg = self.event_neg(emb_fea[1]) # B,T,1
        t_neg = event_neg - torch.unsqueeze(mask_neg, 2) * 1e8 # B,T,1
        att_neg = self.event_softmax(t_neg)
        his_fea_neg = torch.sum(att_neg * emb_fea[1], 1).squeeze(1) # B, H
        his_fea = torch.cat((his_fea_pos, his_fea_neg),dim=-1) # B, 2H

        # distribution
        new_his_fea = F.relu(self.W_s(his_fea))
        mu_i, sigma_i = self.W_mu(new_his_fea), torch.exp(self.W_sigma(new_his_fea))
        epsilon = self.generateEpsilon()
        his_fea = mu_i + epsilon * sigma_i

        output = self.decoder(his_fea) # B, H
        return his_fea_pos, his_fea_neg, output


class MetaNetTwo(torch.nn.Module):
    """our design"""
    def __init__(self, emb_dim, meta_dim):
        super(MetaNetTwo, self).__init__()
        self.emb_dim = emb_dim
        self.event_pos = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))  # character encoder
        self.event_neg = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                             torch.nn.Linear(emb_dim, 1, False))  # character encoder
        self.event_softmax = torch.nn.Softmax(dim=1)

        self.W_s_pos = nn.Linear(emb_dim, emb_dim,bias=False)
        self.W_mu_pos = nn.Linear(emb_dim, emb_dim,bias=False)
        self.W_sigma_pos = nn.Linear(emb_dim, emb_dim,bias=False)

        self.W_s_neg = nn.Linear(emb_dim, emb_dim,bias=False)
        self.W_mu_neg = nn.Linear(emb_dim, emb_dim,bias=False)
        self.W_sigma_neg = nn.Linear(emb_dim, emb_dim,bias=False)

        self.decoder = torch.nn.Sequential(torch.nn.Linear(2*emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim*emb_dim))  # meta-network

    def generateEpsilon(self):
        return torch.randn(size=(self.emb_dim, )).cuda() if config['USE_CUDA'] else torch.randn(size=(self.emb_dim, ))

    def forward(self, emb_fea, seq_index):
        """emb_fea: ([B,T,H],[B,T,H]), seq_index: ([B,T],[B,T])
        return
        """
        # positive encoder
        mask_pos = (seq_index[0] == 0).float() # 0为padding index , B,T, True==1. False==0
        event_pos = self.event_pos(emb_fea[0]) # B,T,1
        t_pos = event_pos - torch.unsqueeze(mask_pos, 2) * 1e8 # B,T,1
        att_pos = self.event_softmax(t_pos)
        his_fea_pos = torch.sum(att_pos * emb_fea[0], 1).squeeze(1) # B, H
        # distribution
        his_fea_pos = F.relu(self.W_s_pos(his_fea_pos))
        mu_i_pos, sigma_i_pos = self.W_mu_pos(his_fea_pos), torch.exp(self.W_sigma_pos(his_fea_pos))
        epsilon = self.generateEpsilon()
        his_fea_pos = mu_i_pos + epsilon * sigma_i_pos

        # negative encoder
        mask_neg = (seq_index[1] == 0).float() # 0为padding index , B,T, True==1. False==0
        event_neg = self.event_neg(emb_fea[1]) # B,T,1
        t_neg = event_neg - torch.unsqueeze(mask_neg, 2) * 1e8 # B,T,1
        att_neg = self.event_softmax(t_neg)
        his_fea_neg = torch.sum(att_neg * emb_fea[1], 1).squeeze(1) # B, H

        # distribution
        his_fea_neg = F.relu(self.W_s_neg(his_fea_neg))
        mu_i_neg, sigma_i_neg = self.W_mu_neg(his_fea_neg), torch.exp(self.W_sigma_neg(his_fea_neg))
        epsilon = self.generateEpsilon()
        his_fea_neg = mu_i_neg + epsilon * sigma_i_neg

        his_fea = torch.cat((his_fea_pos, his_fea_neg), dim=-1) # B, 2H
        # his_fea = his_fea_pos
        output = self.decoder(his_fea) # B, H
        return his_fea_pos, his_fea_neg, output



class MFBasedModelTwo(nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
        super(MFBasedModelTwo, self).__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNetTwo(emb_dim, meta_dim_0)
        self.general_bridge = torch.nn.Linear(emb_dim, emb_dim, False)
        self.triple_loss = TripletLoss(margin=config['margin'])
        self.reg_loss = EmbLoss()
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def proto_loss(self, u_embed, u2centroids_emb):
        """ssl loss"""
        norm_u_embed = F.normalize(u_embed)
        pos_score = torch.mul(norm_u_embed, u2centroids_emb).sum(dim=1)
        pos_score = torch.exp(pos_score/config['ssl_temp'])
        ttl_score = torch.matmul(norm_u_embed, u2centroids_emb.transpose(0,1))
        ttl_score = torch.exp(ttl_score/config['ssl_temp']).sum(dim=1)
        proto_loss = -torch.log(pos_score/ttl_score).sum()
        return proto_loss * config['proto_reg']



    def forward(self, x, stage):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            emb_loss = self.reg_loss(emb)[0]
            return x,  emb_loss
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            emb_loss = self.reg_loss(emb)[0]
            return x,  emb_loss
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x) # B,H
            emb_loss = self.reg_loss(emb)[0]
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x,  emb_loss
        elif stage in ['test_prob']:
            uid_meta = self.src_model.uid_embedding(x[0]) #  [co-user: src_item]
            # random_items = torch.tensor(random.sample(range(0, config["src_tgt_pairs"][config["TASK"]]['iid']), 1000), dtype=torch.long).cuda()
            items_all, items_pop = x[1], x[2]
            iid_meta = self.src_model.iid_embedding(items_all)
            scores = (uid_meta @ iid_meta.T).cpu().numpy() # co_user * items
            items_all = items_all.repeat(uid_meta.shape[0], 1).cpu().numpy() # co_user * items
            items_pop = items_pop.repeat(uid_meta.shape[0], 1).cpu().numpy()
            # ranklists_top = np.take_along_axis(data, np.argsort(-scores), axis=-1)[:, :config['TOPK']] # 
            # ranklists_tail = np.take_along_axis(data, np.argsort(-scores), axis=-1)[:, -config['TOPK']:]
            # return ranklists_top, ranklists_tail
            return scores, items_all, items_pop

        elif stage in ['train_meta', 'test_meta']:
            x, aux_data, (centroids, nodes2cluster) = x
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1)) # B, 1, H

            u2cluster = nodes2cluster[x[:, 0]] # B,
            u2centroids = centroids[u2cluster] # B,e
            if aux_data:
                (top_k, tail_k) = aux_data
                top_sup, tail_sup = top_k(x[:, 0]).long().detach(), tail_k(x[:, 0]).long().detach() # B, K
                positive_item, negative_item = torch.cat([x[:, 2:2+config['his']], top_sup], dim=1), torch.cat([x[:, 2+config['his']:], tail_sup], dim=1)
            else:
                positive_item, negative_item = x[:, 2:2+config['his']], x[:, 2+config['his']:]
            ufea_pos = self.src_model.iid_embedding(positive_item) # positive item embedding B,K+20, H
            ufea_neg = self.src_model.iid_embedding(negative_item) # negative item embedding
            pos_dis, neg_dis, meta_param = self.meta_net.forward((ufea_pos, ufea_neg), (positive_item, negative_item))

            mapping = meta_param.view(-1, self.emb_dim, self.emb_dim) # meta parameter B,H,1 
            uid_emb_bias = torch.bmm(uid_emb_src, mapping) # [B,1, H] * [B,H,1] = B,1,1
            # uid_emb_bias = uid_emb_bias.mean(dim=-1,keepdim=True)
            uid_emb = uid_emb_bias + self.general_bridge(uid_emb_src) # B,1,H
            # uid_emb = uid_emb_bias # B,1,H
            emb = torch.cat([uid_emb, iid_emb], 1)
            # norm_u_embeddings, norm_i_embeddings = reg(emb[:, 0, :]), reg(emb[:, 1, :])
            # output = torch.sum(torch.multiply(emb[:, 0, :], emb[:, 1, :]),
            #                    axis=1, keepdims=False) / (norm_u_embeddings * norm_i_embeddings)
            # output = torch.clamp(output, 1e-6)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1) # mae
            output_i_dis = self.triple_loss(uid_emb.squeeze(dim=1), pos_dis, neg_dis)
            output_u_dis = self.proto_loss(uid_emb.squeeze(dim=1), u2centroids)

            emb_loss = self.reg_loss(emb)[0]
            return output, output_i_dis, output_u_dis, emb_loss,emb
        
        # EMCDR
        elif stage == 'train_map':
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x




class LookupEmbedding_BPR(torch.nn.Module):
    """create embedding lookup table"""
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim) # 从1开始
        nn.init.xavier_uniform_(self.uid_embedding.weight)
        nn.init.xavier_uniform_(self.iid_embedding.weight) # 唯一不同

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1)) # 按照id index emb
        pos_emb = self.iid_embedding(x[:, 1].unsqueeze(1)) # B, 1, H
        neg_emb = self.iid_embedding(x[:, 2].unsqueeze(1)) # B, 1, H

        emb = torch.cat([uid_emb, pos_emb, neg_emb], dim=1)
        return emb

class MFBaseModelBPR(MFBasedModelTwo):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
        super(MFBaseModelBPR, self).__init__(uid_all, iid_all, num_fields, emb_dim, meta_dim_0)
        self.src_model = LookupEmbedding_BPR(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding_BPR(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding_BPR(uid_all, iid_all, emb_dim)
        self.rec_loss = nn.TripletMarginLoss(margin=config['margin'])

    @staticmethod
    def embedding_normalize(embeddings):
        emb_length = torch.sum(embeddings ** 2, dim=1, keepdim=True)
        ones = torch.ones_like(emb_length)
        norm = torch.where(emb_length > 1, emb_length, ones)
        return embeddings / norm

    def forward(self, x, stage):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            emb_loss = self.reg_loss(emb)[0]
            return emb,  emb_loss

        elif stage in ['train_tgt','test_tgt']:
            emb = self.tgt_model.forward(x)
            emb_loss = self.reg_loss(emb)[0]
            return emb,  emb_loss
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            emb_loss = self.reg_loss(emb)[0]
            return emb,  emb_loss

        elif stage in ['train_meta','test_meta']:
            x, aux_data, (centroids, nodes2cluster) = x
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1)) # B, 1, H
            u2cluster = nodes2cluster[x[:, 0]] # B,
            u2centroids = centroids[u2cluster] # B,e
            if aux_data:
                (top_k, tail_k) = aux_data
                top_sup, tail_sup = top_k(x[:, 0]).long().detach(), tail_k(x[:, 0]).long().detach() # B, K
                positive_item, negative_item = torch.cat([x[:, 2:2+config['his']], top_sup], dim=1), torch.cat([x[:, 2+config['his']:], tail_sup], dim=1)
            else:
                positive_item, negative_item = x[:, 2:2+config['his']], x[:, 2+config['his']:]
            ufea_pos = self.src_model.iid_embedding(positive_item) # positive item embedding B,K+20, H
            ufea_neg = self.src_model.iid_embedding(negative_item) # negative item embedding
            pos_dis, neg_dis, meta_param = self.meta_net.forward((ufea_pos, ufea_neg), (positive_item, negative_item))

            mapping = meta_param.view(-1, self.emb_dim, self.emb_dim) # meta parameter B,H,1 
            uid_emb_bias = torch.bmm(uid_emb_src, mapping) # [B,1, H] * [B,H,1] = B,1,1
            uid_emb = uid_emb_bias + self.general_bridge(uid_emb_src) # B,1,H
            emb = torch.cat([uid_emb, iid_emb], 1)
            # norm_u_embeddings, norm_i_embeddings = reg(emb[:, 0, :]), reg(emb[:, 1, :])
            # output = torch.sum(torch.multiply(emb[:, 0, :], emb[:, 1, :]),
            #                    axis=1, keepdims=False) / (norm_u_embeddings * norm_i_embeddings)
            # output = torch.clamp(output, 1e-6)
            # output = self.rec_loss(self.embedding_normalize(emb[:, 0, :]),
            #                        self.embedding_normalize(emb[:, 1, :]),
            #                        self.embedding_normalize(emb[:, 2, :]))
            output_i_dis = self.triple_loss(uid_emb.squeeze(dim=1), pos_dis, neg_dis)
            output_u_dis = self.proto_loss(uid_emb.squeeze(dim=1), u2centroids)
            emb_loss = self.reg_loss(emb)[0]
            return emb, output_i_dis, output_u_dis, emb_loss


class MFBasedModel(nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.reg_loss = EmbLoss()

    def forward(self, x, stage):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            emb_loss = self.reg_loss(emb)[0]
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x,emb_loss
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            emb_loss = self.reg_loss(emb)[0]

            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x,emb_loss
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            emb_loss = self.reg_loss(emb)[0]

            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x, emb_loss
        elif stage in ['train_meta', 'test_meta']: # 这里要改
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1)) # B,1,H
            ufea = self.src_model.iid_embedding(x[:, 2:]) # positive item
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim) # meta parameter
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            emb_loss = self.reg_loss(emb)[0]
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output, emb_loss,emb
        # EMCDR
        elif stage == 'train_map':
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            emb_loss = self.reg_loss(emb)[0]
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x, emb_loss,emb



class GMFBasedModelTwo(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNetTwo(emb_dim, meta_dim)
        self.general_bridge = torch.nn.Linear(emb_dim, emb_dim, False)
        self.triple_loss = TripletLoss(margin=config['margin'])
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.reg_loss = EmbLoss()


    def proto_loss(self, u_embed, u2centroids_emb):
        """ssl loss"""
        norm_u_embed = F.normalize(u_embed)
        pos_score = torch.mul(norm_u_embed, u2centroids_emb).sum(dim=1)
        pos_score = torch.exp(pos_score/config['ssl_temp'])
        ttl_score = torch.matmul(norm_u_embed, u2centroids_emb.transpose(0,1))
        ttl_score = torch.exp(ttl_score/config['ssl_temp']).sum(dim=1)
        proto_loss = -torch.log(pos_score/ttl_score).sum()
        return proto_loss * config['proto_reg']

    def forward(self, x, stage):
        if stage == 'train_src':
            x, emb_loss = self.src_model.forward(x)
            return x, emb_loss
        elif stage in ['train_tgt', 'test_tgt']:
            x,emb_loss = self.tgt_model.forward(x)
            return x,emb_loss
        elif stage in ['train_aug', 'test_aug']:
            x,emb_loss = self.aug_model.forward(x)
            return x,emb_loss

        elif stage in ['test_prob']:
            uid_meta = self.src_model.embedding.uid_embedding(x[0]) 
            items_all, items_pop = x[1], x[2]
            iid_meta = self.src_model.embedding.iid_embedding(items_all)
            scores = (uid_meta @ iid_meta.T).cpu().numpy() # co_user * items
            items_all = items_all.repeat(uid_meta.shape[0], 1).cpu().numpy() # co_user * items
            items_pop = items_pop.repeat(uid_meta.shape[0], 1).cpu().numpy()
            return scores, items_all, items_pop

        elif stage in ['test_meta', 'train_meta']:
            x, aux_data, (centroids, nodes2cluster) = x
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))  # B, 1, H

            u2cluster = nodes2cluster[x[:, 0]]  # B,
            u2centroids = centroids[u2cluster]  # B,e
            if aux_data:
                (top_k, tail_k) = aux_data
                top_sup, tail_sup = top_k(x[:, 0]).long().detach(), tail_k(x[:, 0]).long().detach()  # B, K
                positive_item, negative_item = torch.cat([x[:, 2:2 + config['his']], top_sup], dim=1), torch.cat(
                    [x[:, 2 + config['his']:], tail_sup], dim=1)
            else:
                positive_item, negative_item = x[:, 2:2 + config['his']], x[:, 2 + config['his']:]
            ufea_pos = self.src_model.embedding.iid_embedding(positive_item)  # positive item embedding B,K+20, H
            ufea_neg = self.src_model.embedding.iid_embedding(negative_item)  # negative item embedding
            pos_dis, neg_dis, meta_param = self.meta_net.forward((ufea_pos, ufea_neg), (positive_item, negative_item))
            mapping = meta_param.view(-1, self.emb_dim, self.emb_dim)  # meta parameter B,H,1
            uid_emb_bias = torch.bmm(uid_emb_src, mapping)  # [B,1, H] * [B,H,1] = B,1,1
            uid_emb = uid_emb_bias + self.general_bridge(uid_emb_src)  # B,1,H
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])  # mae
            output_i_dis = self.triple_loss(uid_emb.squeeze(dim=1), pos_dis, neg_dis)
            output_u_dis = self.proto_loss(uid_emb.squeeze(dim=1), u2centroids)
            emb_loss = self.reg_loss(emb)[0]
            return output.squeeze(1), output_i_dis, output_u_dis, emb_loss
        elif stage == 'train_map':
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            emb_loss = self.reg_loss(emb)[0]

            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1),emb_loss


class GMFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.reg_loss = EmbLoss()


    def forward(self, x, stage):
        if stage == 'train_src':
            x,emb_loss = self.src_model.forward(x)
            return x,emb_loss
        elif stage in ['train_tgt', 'test_tgt']:
            x,emb_loss = self.tgt_model.forward(x)
            return x,emb_loss
        elif stage in ['train_aug', 'test_aug']:
            x ,emb_loss= self.aug_model.forward(x)
            return x,emb_loss
        elif stage in ['test_meta', 'train_meta']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)) # B, 1, H
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            emb_loss = self.reg_loss(emb)[0]
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return output.squeeze(1),emb_loss
        elif stage == 'train_map':
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1),0




class DNNBasedModelTwo(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNetTwo(emb_dim, meta_dim)
        self.general_bridge = torch.nn.Linear(emb_dim, emb_dim, False)
        self.triple_loss = TripletLoss(margin=config['margin'])
        self.reg_loss = EmbLoss()
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def proto_loss(self, u_embed, u2centroids_emb):
        """ssl loss"""
        norm_u_embed = F.normalize(u_embed)
        pos_score = torch.mul(norm_u_embed, u2centroids_emb).sum(dim=1)
        pos_score = torch.exp(pos_score/config['ssl_temp'])
        ttl_score = torch.matmul(norm_u_embed, u2centroids_emb.transpose(0,1))
        ttl_score = torch.exp(ttl_score/config['ssl_temp']).sum(dim=1)
        proto_loss = -torch.log(pos_score/ttl_score).sum()
        return proto_loss * config['proto_reg']

    def forward(self, x, stage):
        if stage == 'train_src':
            x ,emb_loss= self.src_model.forward(x)
            return x,emb_loss
        elif stage in ['train_tgt', 'test_tgt']:
            x ,emb_loss= self.tgt_model.forward(x)
            return x,emb_loss
        elif stage in ['train_aug', 'test_aug']:
            x ,emb_loss= self.aug_model.forward(x)
            return x,emb_loss

        elif stage in ['test_prob']:
            uid_meta = self.src_model.embedding.uid_embedding(x[0]) # 和源域中的item [co-user: src_item]
            items_all, items_pop = x[1], x[2]
            iid_meta = self.src_model.embedding.iid_embedding(items_all)
            scores = (uid_meta @ iid_meta.T).cpu().numpy() # co_user * items
            items_all = items_all.repeat(uid_meta.shape[0], 1).cpu().numpy() # co_user * items
            items_pop = items_pop.repeat(uid_meta.shape[0], 1).cpu().numpy()
            return scores, items_all, items_pop

        elif stage in ['test_meta', 'train_meta']:
            x, aux_data, (centroids, nodes2cluster) = x
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))  # B, 1, H
            u2cluster = nodes2cluster[x[:, 0]]  # B,
            u2centroids = centroids[u2cluster]  # B,e
            if aux_data:
                (top_k, tail_k) = aux_data
                top_sup, tail_sup = top_k(x[:, 0]).long().detach(), tail_k(x[:, 0]).long().detach()  # B, K
                positive_item, negative_item = torch.cat([x[:, 2:2 + config['his']], top_sup], dim=1), torch.cat(
                    [x[:, 2 + config['his']:], tail_sup], dim=1)
            else:
                positive_item, negative_item = x[:, 2:2 + config['his']], x[:, 2 + config['his']:]
            ufea_pos = self.src_model.embedding.iid_embedding(positive_item)  # positive item embedding B,K+20, H
            ufea_neg = self.src_model.embedding.iid_embedding(negative_item)  # negative item embedding
            pos_dis, neg_dis, meta_param = self.meta_net.forward((ufea_pos, ufea_neg), (positive_item, negative_item))
            mapping = meta_param.view(-1, self.emb_dim, self.emb_dim)  # 
            uid_emb_bias = torch.bmm(uid_emb_src, mapping)  # [B,1, H] * [B,H,1] = B,1,1
            uid_emb = uid_emb_bias + self.general_bridge(uid_emb_src)  # B,1,H
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)  # mae
            output_i_dis = self.triple_loss(uid_emb.squeeze(dim=1), pos_dis, neg_dis)
            output_u_dis = self.proto_loss(uid_emb.squeeze(dim=1), u2centroids)
            emb_loss = self.reg_loss(emb)[0]
            return output, output_i_dis, output_u_dis, emb_loss

        elif stage == 'train_map':
            src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            emb_loss = self.reg_loss(emb)[0]
            return x,emb_loss


class DNNBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.reg_loss = EmbLoss()

    def forward(self, x, stage):
        if stage == 'train_src':
            x ,emb_loss= self.src_model.forward(x)
            return x,emb_loss
        elif stage in ['train_tgt', 'test_tgt']:
            x ,emb_loss= self.tgt_model.forward(x)
            return x,emb_loss
        elif stage in ['train_aug', 'test_aug']:
            x ,emb_loss= self.aug_model.forward(x)
            return x,emb_loss
        elif stage in ['test_meta', 'train_meta']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            emb_loss = self.reg_loss(emb)[0]
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return output, emb_loss
        elif stage == 'train_map':
            src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            emb_loss = self.reg_loss(emb)[0]
            return x,emb_loss

