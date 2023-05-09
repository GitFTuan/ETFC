#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/26 21:04
# @Author : fhh
# @FileName: model.py
# @Software: PyCharm

from util import *
import torch
import torch.nn as nn


class StudentModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads):
        super(StudentModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout
        self.fan_epoch = fan_epoch
        self.num_heads = num_heads
        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)

        self.BiRNN = nn.LSTM(input_size=self.embedding_size,
                             hidden_size=self.embedding_size // 2,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True)

        self.attention_encode = MASK_AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
        self.transformer = transformer_encode(self.dropout, self.embedding_size, self.num_heads)
        self.Attention = AdditiveAttention(input_size=self.embedding_size,
                                           value_size=21,
                                           num_hiddens=self.embedding_size,
                                           dropout=0.5)

        self.full6 = nn.Linear(2100, 1000)
        self.full1 = nn.Linear(9600, 4032)
        self.bn1 = nn.BatchNorm1d(4032)
        self.full2 = nn.Linear(4032, 2304)
        self.bn = nn.BatchNorm1d(2304)
        self.full3 = nn.Linear(2304, 1000)
        self.full4 = nn.Linear(1000, 500)
        self.full5 = nn.Linear(500, 256)

        self.Flatten = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.output_size)
        self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self, train_data, valid_lens, in_feat=False):
        """---------------------氨基酸编码------------------------"""
        embed_output = self.embedding(train_data)
        '''-----------------------------------------------------'''

        vectors = embed_output

        '''----------------------attention----------------------'''
        attention_encode = vectors
        for i in range(self.fan_epoch):
            attention_encode = self.attention_encode(attention_encode, valid_lens)

        attention_output, weights = self.Attention(attention_encode, attention_encode, attention_encode, valid_lens)
        '''-----------------------------------------------------'''
        out = attention_output.contiguous().view(attention_output.size()[0], -1)

        # 全连接层
        label = self.full2(out)
        label = self.bn(label)
        label = torch.nn.ReLU()(label)

        label = self.full3(label)
        label = torch.nn.ReLU()(label)

        label2 = self.full4(label)
        label = torch.nn.ReLU()(label2)

        label3 = self.full5(label)
        label = torch.nn.ReLU()(label3)

        label4 = self.Flatten(label)
        label = torch.nn.ReLU()(label4)
        out_label = self.out(label)

        if in_feat:
            return label2, label3, label4, out_label
        else:
            return out_label


class ETFC(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout, fan_epoch, num_heads, max_pool=5):
        super(ETFC, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout
        self.fan_epoch = fan_epoch
        self.num_heads = num_heads
        self.max_pool = max_pool
        if max_pool == 2:
            shape = 6016
        elif max_pool == 3:
            shape = 3968
        elif max_pool == 4:
            shape = 2944
        elif max_pool == 5:
            shape = 2304
        else:
            shape = 1920
        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.pos_encoding = PositionalEncoding(num_hiddens=self.embedding_size, dropout=self.dropout)
        self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )
        self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )
        self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=1
                                     )
        self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1
                                     )
        # 序列最短为5，故将卷积核分别设为：2、3、4、5
        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=self.max_pool)

        self.attention_encode = AttentionEncode(self.dropout, self.embedding_size, self.num_heads)
        self.fan = FAN_encode(self.dropout, shape)

        self.full3 = nn.Linear(shape, 1000)
        self.full4 = nn.Linear(1000, 500)
        self.full5 = nn.Linear(500, 256)
        # self.full6 = nn.Linear(4608, 2304)
        self.Flatten = nn.Linear(256, 64)
        self.out = nn.Linear(64, self.output_size)
        self.dropout = torch.nn.Dropout(self.dropout)

    def TextCNN(self, x):
        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        x1 = self.MaxPool1d(x1)

        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        x2 = self.MaxPool1d(x2)

        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        x3 = self.MaxPool1d(x3)

        x4 = self.conv4(x)
        x4 = torch.nn.ReLU()(x4)
        x4 = self.MaxPool1d(x4)

        y = torch.cat([x1, x2, x3, x4], dim=-1)

        x = self.dropout(y)

        x = x.view(x.size(0), -1)

        return x

    def forward(self, train_data, valid_lens, in_feat=False):

        embed_output = self.embed(train_data)

        '''----------------------位置编码------------------------'''
        pos_output = self.pos_encoding(self.embed(train_data) * math.sqrt(self.embedding_size))
        '''-----------------------------------------------------'''

        '''----------------------attention----------------------'''
        attention_output = self.attention_encode(pos_output)
        '''-----------------------------------------------------'''

        '''----------------------特征相加-------------------------'''
        vectors = embed_output + attention_output
        '''------------------------------------------------------'''

        '''---------------------data_cnn-----------------------'''
        cnn_input = vectors.permute(0, 2, 1)
        cnn_output = self.TextCNN(cnn_input)
        '''-----------------------------------------------------'''

        '''---------------------fan_encode----------------------'''
        fan_encode = cnn_output.unsqueeze(0).permute(1, 0, 2)
        for i in range(self.fan_epoch):
            fan_encode = self.fan(fan_encode)
        '''-----------------------------------------------------'''

        out = fan_encode.squeeze()

        # 全连接层
        label = self.full3(out)
        label = torch.nn.ReLU()(label)
        label1 = self.full4(label)
        label = torch.nn.ReLU()(label1)
        label2 = self.full5(label)
        label = torch.nn.ReLU()(label2)
        label3 = self.Flatten(label)
        label = torch.nn.ReLU()(label3)
        out_label = self.out(label)

        if in_feat:
            return label1, label2, label3, out_label
        else:
            return out_label
