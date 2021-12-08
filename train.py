#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import os
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

from metrics import cal_auc
from utils import data_preprocess
from net import DeepFM
import data_process

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def run_trainer(args):
    user_file, movie_file, rating_file = args.users_data, args.movies_data, args.ratings_data
    data = data_preprocess(user_file, movie_file, rating_file)
    train_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8, random_state=0)
    training_set = data_process.JoinedDataset(
        train_data, args.single_feature_col, args.multi_feature_col, args.label_col)
    testing_set = data_process.JoinedDataset(
        test_data, args.single_feature_col, args.multi_feature_col, args.label_col)
    training_params = {
        'batch_size': args.batch,
        'num_workers': 6,
        'drop_last': True,
        'shuffle': False,
    }
    testing_params = {
        'batch_size': 512,
        'num_workers': 6,
        'pin_memory': True,
        'drop_last': False,
    }
    net = DeepFM()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        # 配置每个进程的gpu
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.distributed.init_process_group(backend='nccl')
        train_sampler = DistributedSampler(training_set)
        test_sampler = DistributedSampler(testing_set)
        training_params['sampler'] = train_sampler
        testing_params['sampler'] = test_sampler
        net.to(device)
    training_generator = DataLoader(training_set, **training_params)
    testing_generator = DataLoader(testing_set, **testing_params)

    if torch.cuda.device_count() > 1:  # 多卡
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank)
    # Setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # 如果是接着训练的话，需要把上次的训练结果导入，然后继续进行训练
    if args.base_optimizer is not None:
        ckpt = torch.load(args.base_optimizer)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = args.lr
        del ckpt
    criterion = nn.BCELoss()

    train_params = {
        'net': net,
        'optimizer': optimizer,
        'criterion': criterion,
        'epochs': args.epoch,
        'training_generator': training_generator,
        'testing_generator': testing_generator,
        'model_name': os.path.join(args.output, 'model.bin'),
        'optimizer_name': os.path.join(args.output, 'optimizer.pkl'),
    }
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    train(**train_params)


def test(net, testing_generator):
    net.eval()
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for i, batch in enumerate(testing_generator):
            for k in batch:
                if not isinstance(batch[k], list):
                    batch[k] = batch[k].to(device, non_blocking=True)
            outs = net.module.predict(batch) if device != 'cpu' else net.predict(batch)
            for uid, label, predict in zip(
                    batch['uid'].tolist(), batch['label'].tolist(), outs.tolist()):
                results.append((uid, label, predict))
    net.train()
    return results


def train(net,
          optimizer,
          criterion,
          epochs,
          training_generator,
          testing_generator,
          model_name,
          optimizer_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for epoch in range(epochs):
        for i, batch in enumerate(training_generator):
            if i != 0 and i % 5000 == 0:
                results = test(net, testing_generator)
                total_auc, uid_auc = cal_auc(results)
                logging.info(f"test total auc is: {total_auc}, test uid auc is: {uid_auc}")
            if device == 'cuda':
                for k in batch:
                    if not isinstance(batch[k], list):
                        batch[k] = batch[k].to('cuda', non_blocking=True)
            optimizer.zero_grad()
            output = net(batch)
            labels = batch['label'].unsqueeze(-1).float()
            labels = labels.to(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            logging.info(f'Epoch {epoch},{i} Loss {loss.item()}')
        MODEL_SAVE_PATH = f'{model_name}.{epoch}'
        OPTIMIZER_SAVE_PATH = f'{optimizer_name}.{epoch}'
        torch.save({
            'model_state_dict': net.state_dict(),
        }, MODEL_SAVE_PATH)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
        }, OPTIMIZER_SAVE_PATH)
        results = test(net, testing_generator)
        total_auc, uid_auc = cal_auc(results)
        logging.info(f"test total auc is: {total_auc}, test uid auc is: {uid_auc}")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--users_data", default='movielens/users.dat')
    arg_parser.add_argument("--movies_data", default='movielens/movies.dat')
    arg_parser.add_argument("--ratings_data", default='movielens/ratings.dat')
    arg_parser.add_argument("--single_feature_col", default=["uid", "movieid", "gender", "age", "occ", "zip_code"])
    arg_parser.add_argument("--multi_feature_col", default=["genres"])
    arg_parser.add_argument("--label_col", default="label")
    arg_parser.add_argument('--lr', type=float, help='learning rate', default=0.005)
    arg_parser.add_argument('--batch', type=int, help='batch_size', default=2048)
    arg_parser.add_argument('--epoch', type=int, default=4)
    arg_parser.add_argument('--output', help='output model path', default='./model/')
    arg_parser.add_argument('--base_model', nargs='?', default=None, help='use for fine tune')
    arg_parser.add_argument('--base_optimizer', nargs='?', default=None, help='use for fine tune')
    arg_parser.add_argument('--local_rank', default=-1, type=int,
                            help='node local rank for distributed training')
    args = arg_parser.parse_args()
    run_trainer(args)


if __name__ == '__main__':
    main()
