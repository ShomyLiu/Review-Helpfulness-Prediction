# -*- encoding: utf-8 -*-
import time
import random
import fire
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ReviewData
from framework import Model
import models
import config

import logging


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    return zip(*batch)


def unpack_input(opt, datas):

    x, rating_score, is_helpful, helpful_score = datas
    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    reviews = [opt.pair_review_dict[(u, i)] for u, i in zip(uids, iids)]
    review2len = [opt.pair_review2len_dict[(u, i)] for u, i in zip(uids, iids)]

    data = [uids, iids, reviews, review2len]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))

    scores = [is_helpful, helpful_score]
    scores = list(map(lambda x: torch.FloatTensor(x).cuda(), scores))
    return data, scores[0], scores[1]


def train(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Gourmet_Food_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    logging.basicConfig(filename=f"logs/{opt}.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    logging.info('{}: train data: {}; val data: {}'.format(now(), len(train_data), len(val_data)))

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # training
    logging.info("start training....")
    min_loss = 1e+20
    best_auc = -1.
    best_per = -1.
    best_epoch = 0
    cre_loss = nn.BCEWithLogitsLoss()
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        model.train()
        for idx, datas in enumerate(train_data_loader):
            train_datas, is_helpful, helpful_score = unpack_input(opt, datas)
            optimizer.zero_grad()
            output = model(train_datas)
            loss = cre_loss(output, is_helpful.float())
            cur_loss = loss.item()
            total_loss += cur_loss
            loss.backward()
            optimizer.step()

        scheduler.step(epoch)
        logging.info(f"{now()}: epoch {epoch}: total_loss: {total_loss}")
        print(f"epoch: {epoch}")
        auc, corr, predict_loss = predict(model, val_data_loader, opt, logging)
        if predict_loss < min_loss:
            min_loss = predict_loss
        if auc > best_auc:
            model.save(name=opt.dataset, epoch=epoch, opt=f"{opt}")
            best_epoch = epoch
            best_auc = auc
            best_per = corr
            logging.info("model save")

    logging.info("----"*20)
    logging.info(f"{now()}:{opt.model}:{opt} \n\t\t best_auc:{best_auc}, best_per:{best_per}")
    logging.info("----"*20)
    print("----"*20)
    print(f"{now()}:{opt.model}:{opt} \n\t epoch:{best_epoch}: best_auc:{best_auc}, best_per:{best_per}")
    print("----"*20)


def test(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Gourmet_Food_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    logging.basicConfig(filename=f"logs/{opt}.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model)).cuda()
    print("load...")
    model.load("./checkpoints/DPHP_Gourmet_Food_data_cfg-Gourmet_Food_data-poolatt-lr0.001-wd0.0005-drop0.1-id32-hidden100.pth")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    auc, corr, predict_loss = predict(model, test_data_loader, opt, logging)


def predict(model, test_data_loader, opt, logging):

    eval_loss = 0.0
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for idx, datas in enumerate(test_data_loader):
            test_datas, is_helpful, helpful_score = unpack_input(opt, datas)
            cls = model(test_datas).squeeze()
            loss = F.binary_cross_entropy_with_logits(cls, is_helpful)
            eval_loss += loss.item()
            labels.extend(is_helpful.cpu().numpy().tolist())
            preds.extend(cls.cpu().numpy().tolist())

    model.train()
    auc = roc_auc_score(labels, preds)
    corr, p_value = pearsonr(labels, preds)
    corr, p_value, auc = round(corr, 4), round(p_value, 4), round(auc, 4)
    logging.info(f"\ttest loss: {eval_loss}, AUC: {auc}, Pearsonr: {corr}")
    print(f"\ttest loss: {eval_loss}, AUC: {auc}, Pearsonr: {corr}")

    return auc, corr, eval_loss


if __name__ == "__main__":
    fire.Fire()
