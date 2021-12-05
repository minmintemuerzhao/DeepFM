#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import embedding_dim
from utils import embedding_num


class feature_embedding(nn.Module):
    def __init__(self, device='cpu'):
        super(feature_embedding, self).__init__()
        self.device = device
        self.uid_embs = self.init_embedding('uid')
        self.movieid_embs = self.init_embedding('movieid')
        self.gender_embs = self.init_embedding('gender')
        self.age_embs = self.init_embedding('age')
        self.occ_embs = self.init_embedding('occ')
        self.zip_code_embs = self.init_embedding('zip_code')
        self.genres_embs = self.init_embedding('genres')

    def init_embedding(self, feature_type, padding=True):
        if padding:
            emb = nn.Embedding(
                embedding_num[feature_type],
                embedding_dim[feature_type],
                padding_idx=0
            )
            return emb.to(self.device, non_blocking=True)
        else:
            emb = nn.Embedding(
                embedding_num[feature_type],
                embedding_dim[feature_type],
            )
            return emb.to(self.device, non_blocking=True)

    def forward(self, batch):
        return {
            'uid': self.uid_embs(batch['uid']),
            'movieid': self.movieid_embs(batch['movieid']),
            'gender': self.gender_embs(batch['gender']),
            'age': self.age_embs(batch['age']),
            'occ': self.occ_embs(batch['occ']),
            'zip_code': self.zip_code_embs(batch['zip_code']),
            'genres': torch.mean(self.genres_embs(batch['genres']), dim=-2),
        }


class DeepFM(nn.Module):
    def __init__(self, device='cpu'):
        super(DeepFM, self).__init__()

        # 定好随机种子, 保证每次重新运行时，相同的输入得到相同的输出
        torch.manual_seed(0)
        if device != 'cpu':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.embeddings = feature_embedding(device)
        self.w = torch.ones((sum(embedding_num.values()), 1), device=device)
        self.b = torch.zeros((1,), device=device)
        # self.w = torch.ones((sum(embedding_num.values()), 1))
        # self.b = torch.zeros((1,))
        self.hidden_units = [64, 32, 1]
        self.sigmoid = nn.Sigmoid()
        self.dnn = self.dnn(sum(embedding_dim.values()), self.hidden_units)

    def dnn(self, input_dim, hidden_units, use_bn=False, dropout_rate=0):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        for i, h in enumerate(hidden_units[:-1]):
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            else:
                layers.append(nn.PReLU())
            if dropout_rate != 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(h, hidden_units[i + 1]))
        self.network = nn.Sequential(*layers)
        return self.network


    def forward(self, batch, is_debug=False):
        # 一阶
        one_stage = torch.matmul(batch['data_vector'], self.w) + self.b
        # 二阶
        embedding_result = self.embeddings(batch)
        # cat起来，并行计算
        embedding_cat = torch.cat([v.unsqueeze(1) for _, v in embedding_result.items()], dim=1)
        # square sum part
        squared_feature_embs = torch.square(embedding_cat)
        squared_sum_feature_embs = torch.sum(squared_feature_embs, dim=-1)
        # sum square part
        sum_feature_embs = torch.sum(embedding_cat, dim=-1)
        sum_square_feature_embs = torch.square(sum_feature_embs)
        two_stage = (1 / 2) * torch.sum(
            torch.sub(sum_square_feature_embs, squared_sum_feature_embs), dim=-1, keepdim=True)
        fm_stage = one_stage + two_stage
        # dnn
        embedding_concat = torch.cat(list(embedding_result.values()), dim=-1)
        result = self.network(embedding_concat)
        return self.sigmoid(one_stage + fm_stage + result)

    def predict(self, batch):
        outs = self.forward(batch)

        if len(outs.shape) == 1:  # when predict size is 1
            outs = outs.unsqueeze(0)
        return outs