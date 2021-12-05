#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sklearn import metrics


def cal_group_auc(datas):
    uid_auc_list = []
    for _, data_list in datas:
        labels = []
        scores = []
        for label, score in data_list:
            labels.append(label)
            scores.append(score)
        uid_auc = metrics.roc_auc_score(labels, scores)
        uid_auc_list.append(uid_auc)
    return sum(uid_auc_list) / len(uid_auc_list) if uid_auc_list != 0 else -1


def cal_auc(results):
    """返回基于样本粒度的auc以及基于uid的groud auc."""
    y_label = []
    y_predict = []

    uid_datas = {}
    for uid, label, predict in results:
        y_label.append(label)
        y_predict.append(predict)

        if uid not in uid_datas:
            uid_datas[uid] = []
        uid_datas[uid].append((label, predict))

    sample_auc = metrics.roc_auc_score(y_label, y_predict)
    uid_auc = cal_group_auc(uid_datas)
    return sample_auc, uid_auc
