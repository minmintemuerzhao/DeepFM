#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torch.utils import data

from utils import embedding_num

class JoinedDataset(data.Dataset):
    def __init__(self,
                 data,
                 single_feature_col,
                 multi_feature_col,
                 label_col):
        super(JoinedDataset, self).__init__()
        self.data = data
        self.single_feature_col = single_feature_col
        self.multi_feature_col = multi_feature_col
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        result = {}
        # 初始化一个向量，用于表示各个特征的取值情况
        data_vector = torch.zeros((sum(embedding_num.values()),))
        data_vector_index = 0
        for col in self.single_feature_col:
            feature_val = self.data.iloc[index][col]
            result[col] = feature_val
            # 索引需要减1
            data_vector[data_vector_index + feature_val - 1] = 1
            data_vector_index += embedding_num[col]
        for col in self.multi_feature_col:
            result[col] = torch.tensor(
                self.data.iloc[index][col] + [0] * (embedding_num[col] - len(self.data.iloc[index][col])))
            # 索引需要减1
            data_vector[data_vector_index + torch.tensor(self.data.iloc[index][col]) - 1] = 1
            data_vector_index += embedding_num[col]
        result[self.label_col] = torch.tensor(self.data.iloc[index][self.label_col])
        result['data_vector'] = data_vector
        return result


