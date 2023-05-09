#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/28 15:20
# @Author : fhh
# @FileName: KD_main.py
# @Software: PyCharm

import time
import datetime
import os
import csv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from loss_functions import *
from model import *
from train import evaluate, CosineScheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']


def PadEncode(data, label, max_len):  # 序列编码
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0
        if length <= max_len and sign == 0:
            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e), np.array(seq_length)


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label


def main(num, data):
    first_dir = 'dataset'

    max_length = 50  # the longest length of the peptide sequence

    # getting train data and test data
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

    # Converting the list collection to an array
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)

    # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
    x_train, y_train, train_length = PadEncode(train_sequence_data, y_train, max_length)
    x_test, y_test, test_length = PadEncode(test_sequence_data, y_test, max_length)

    x_train = torch.LongTensor(x_train)  # torch.Size([7872, 50])
    x_test = torch.LongTensor(x_test)  # torch.Size([1969, 50])
    train_length = torch.LongTensor(train_length)

    y_test = torch.Tensor(y_test)
    y_train = torch.Tensor(y_train)
    test_length = torch.LongTensor(test_length)

    """Create a dataset and split it"""
    dataset_train = list(zip(x_train, y_train, train_length))
    dataset_test = list(zip(x_test, y_test, test_length))
    dataset_train = DataLoader(dataset_train, batch_size=data['batch_size'], shuffle=True, pin_memory=True)
    dataset_test = DataLoader(dataset_test, batch_size=data['batch_size'], shuffle=True, pin_memory=True)

    PATH = os.getcwd()
    each_model = os.path.join(PATH, 'result', 'Model', 'data', 'kd_data' + str(num) + '.h5')
    torch.save(dataset_test, each_model)
    # 设置训练参数
    vocab_size = 50
    output_size = 21
    heads = 8

    # 初始化参数训练模型相关参数
    Teacher_model = ETFC(vocab_size, 192, output_size, 0.6, 1, 8).to(DEVICE)
    model_path = './dataset/Model/teacher/tea_model.pth'
    Teacher_model.load_state_dict(torch.load(model_path))
    Teacher_rate_learning = data['learning_rate']
    Teacher_model.eval()

    Student_lr_scheduler = CosineScheduler(10000, base_lr=Teacher_rate_learning, warmup_steps=500)
    Student_model = StudentModel(vocab_size, data['embedding_size'], output_size, data['dropout'], data['fan_epochs'],
                                 heads).to(DEVICE)
    Student_optimizer = torch.optim.Adam(Student_model.parameters(), lr=Teacher_rate_learning)

    criterionFDL = FocalDiceLoss()
    criterionCEL = torch.nn.BCEWithLogitsLoss()
    temp = 1
    alpha = 0.7

    step = 1
    print("知识蒸馏")
    for epoch in range(data["epochs"]):
        Student_model.train()
        start = time.time()
        total_loss = 0
        # 训练集上训练模型权重
        for data_train, targets, train_length in dataset_train:
            data_train = data_train.to(DEVICE)
            targets = targets.to(DEVICE)
            train_length = train_length.to(DEVICE)
            # 教师模型预测
            with torch.no_grad():
                out1_t, out2_t, out3_t, out_t = Teacher_model(data_train.long(), train_length, in_feat=True)
                soft_label = nn.Sigmoid()(out_t.detach() / temp)

            # 学生模型预测
            out1_s, out2_s, out3_s, out_s = Student_model(data_train.long(), train_length, in_feat=True)

            # 计算hard_loss和kd_loss
            stu_loss = criterionFDL(out_s, targets)
            kd_loss = criterionCEL(out_s / temp, soft_label.detach())

            # 将hard_loss和kd_loss加权求和
            loss = stu_loss + kd_loss * alpha

            # 反向传播, 优化权重
            Student_optimizer.zero_grad()
            loss.backward()
            Student_optimizer.step()
            for param_group in Student_optimizer.param_groups:
                param_group['lr'] = Student_lr_scheduler(step)
            total_loss += loss.item()
            step += 1

        finish = time.time()
        print(f'[ Epoch {epoch + 1} ', end='')
        print("运行时间{}s".format(finish - start))
        print(f'loss={total_loss / len(dataset_train)} ]')
    test_score = evaluate(Student_model, dataset_test, device=DEVICE)
    "-------------------------------------------保存模型参数-----------------------------------------------"
    PATH = os.getcwd()
    each_model = os.path.join(PATH, 'result', 'Model', 'student', 'KD_stu_model' + str(num) + '.pth')
    torch.save(Student_model.state_dict(), each_model, _use_new_zipfile_serialization=False)
    "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------输出模型结果-----------------------------------------------"
    # print(f"runtime:{runtime:.3f}s")
    print("测试集：")
    print(f'aiming: {test_score["aiming"]:.3f}')
    print(f'coverage: {test_score["coverage"]:.3f}')
    print(f'accuracy: {test_score["accuracy"]:.3f}')
    print(f'absolute_true: {test_score["absolute_true"]:.3f}')
    print(f'absolute_false: {test_score["absolute_false"]:.3f}')
    "---------------------------------------------------------------------------------------------------"

    "-------------------------------------------保存模型结果-----------------------------------------------"
    title = ['Model', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'RunTime', 'Test_Time']

    model_name = "KD_stu_test"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = [[model_name, '%.3f' % test_score["aiming"],
                '%.3f' % test_score["coverage"],
                '%.3f' % test_score["accuracy"],
                '%.3f' % test_score["absolute_true"],
                '%.3f' % test_score["absolute_false"],
                now]]

    path = "{}/{}.csv".format('result', 'KD_stu')

    if os.path.exists(path):
        data1 = pd.read_csv(path, header=None)
        one_line = list(data1.iloc[0])
        if one_line == title:
            with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
                writer = csv.writer(t)  # 这一步是创建一个csv的写入器
                writer.writerows(content)  # 写入样本数据
        else:
            with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
                writer = csv.writer(t)  # 这一步是创建一个csv的写入器
                writer.writerow(title)  # 写入标签
                writer.writerows(content)  # 写入样本数据
    else:
        with open(path, 'a+', newline='') as t:  # numline是来控制空的行数的
            writer = csv.writer(t)  # 这一步是创建一个csv的写入器

            writer.writerow(title)  # 写入标签
            writer.writerows(content)  # 写入样本数据
    "---------------------------------------------------------------------------------------------------"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    clip_pos = 0.7
    clip_neg = 0.5
    # pos_weight = 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    pos_weight = 0.3

    batch_size = 192
    epochs = 200
    learning_rate = 0.0018

    embedding_size = 192
    dropout = 0.6
    fan_epochs = 1
    num_heads = 8
    para = {'clip_pos': clip_pos,
            'clip_neg': clip_neg,
            'pos_weight': pos_weight,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'embedding_size': embedding_size,
            'dropout': dropout,
            'fan_epochs': fan_epochs}
    for i in range(10):
        main(i, para)
