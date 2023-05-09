#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/28 17:13
# @Author  : fhh
# @File    : train.py
# @Software: PyCharm
import time
import torch
import math
import numpy as np
from sklearn import metrics
from torchinfo import summary
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
import evaluation
import matplotlib.pyplot as plt


def evaluate(model, datadl, device="cpu"):
    model.to(device)
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y, z in datadl:
            x = x.to(device).long()
            y = y.to(device).long()
            predictions.extend(model(x, z).tolist())
            labels.extend(y.tolist())

    scores = evaluation.evaluate(np.array(predictions), np.array(labels))
    return scores


def scoring(y_true, y_score):
    threshold = 0.5
    y_pred = [int(i >= threshold) for i in y_score]
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix.flatten()
    sen = tp / (fn + tp)
    spe = tn / (fp + tn)
    pre = metrics.precision_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_score)
    pr, rc, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(rc, pr)
    f1 = metrics.f1_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    return dict(SEN=sen, SPE=spe, PRE=pre, F1=f1, MCC=mcc, ACC=acc, AUC=auc, AUPR=aupr, TN=tn, FP=fp, FN=fn, TP=tp)


class DataTrain:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", *args, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler

        self.device = device
        self.model.to(self.device)

    def train_step(self, train_iter, epochs=None, plot_picture=False):
        x_plot = []
        y_plot = []
        epochTrainLoss = []
        # for train_data, train_label in train_iter:
        #     train_data, train_label = train_data.to(self.device), train_label.to(self.device)
        #     summary(self.model, train_data.shape, dtypes=['torch.IntTensor'])
        #     break
        steps = 1
        for epoch in range(1, epochs+1):
            # metric = Accumulator(2)
            start = time.time()
            total_loss = 0
            for train_data, train_label, train_length in train_iter:
                self.model.train()  # 进入训练模式

                train_data, train_label, train_length = train_data.to(self.device), train_label.to(self.device), train_length.to(self.device)
                y_hat = self.model(train_data.long(), train_length)
                loss = self.criterion(y_hat, train_label.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                x_plot.append(epoch)
                y_plot.append(self.lr_scheduler(epoch))
                total_loss += loss.item()
                steps += 1
            # train_loss = metric[0] / metric[1]
            # x_plot.append(self.lr_scheduler.last_epoch)
            # y_plot.append(self.lr_scheduler.get_lr()[0])
            # x_plot.append(epoch)
            # y_plot.append(self.lr_scheduler(epoch))
            # epochTrainLoss.append(train_loss)
            finish = time.time()

            print(f'[ Epoch {epoch} ', end='')
            print("运行时间{}s".format(finish - start))
            print(f'loss={total_loss / len(train_iter)} ]')

        if plot_picture:
            # 绘制学习率变化曲线
            plt.plot(x_plot, y_plot, 'r')
            plt.title('lr value of LambdaLR with (Cos_warmup) ')
            plt.xlabel('step')
            plt.ylabel('lr')
            plt.savefig('./result/Cos_warmup.jpg')
            # plt.show()

            # # 绘制损失函数曲线
            # plt.figure()
            # plt.plot(x_plot, epochTrainLoss)
            # plt.legend(['trainLoss'])
            # plt.xlabel('epochs')
            # plt.ylabel('SHLoss')
            # plt.savefig('./image/Cos_warmup_Loss(200).jpg')
            # # plt.show()

    def KD_step(self, train_iter, epochs=None, plot_picture=False):
        steps = 1
        for epoch in range(1, epochs+1):
            # metric = Accumulator(2)
            start = time.time()
            total_loss = 0
            for train_data, train_label in train_iter:
                self.model.train()  # 进入训练模式

                train_data, train_label = train_data.to(self.device), train_label.to(self.device)
                y_hat = self.model(train_data.long())
                loss = self.criterion(y_hat, train_label.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                # x_plot.append(epoch)
                # y_plot.append(self.lr_scheduler(epoch))
                total_loss += loss.item()
                steps += 1
            finish = time.time()

            print(f'[ Epoch {epoch} ', end='')
            print("运行时间{}s".format(finish - start))
            print(f'loss={total_loss / len(train_iter)} ]')


def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch-1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch-1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
