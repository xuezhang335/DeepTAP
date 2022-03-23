import os
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
aa_idx = {"X": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
          "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20}


class Attention(nn.Module):
    def __init__(self, atten_size):
        # atten_size = [hidden_size * n_bidirection]
        super(Attention, self).__init__()
        self.w = nn.Parameter(torch.rand(atten_size, 1))
        self.b = nn.Parameter(torch.zeros(atten_size))

    def forward(self, x):
        # x = [seq_len, batch_size, hidden_size * n_bidirection]
        dot = torch.matmul(x, self.w) + self.b  # [seq_len, batch_size, 1]
        dot = torch.tanh(dot)  # [seq_len, batch_size, 1]
        alpha = torch.softmax(dot, dim=0)  # [seq_len, batch_size, 1]
        # [seq_len, batch_size, hidden_size * n_bidirection]
        output = x * alpha
        output = output.sum(dim=0)  # [batch_]size, hidden_size * n_bidirection
        return output


class Model(pl.LightningModule):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model_type = config["model_type"]
        self.batch_size = config["batch_size"]
        self.dropout = config["dropout"]
        self.bidirection = config["bidirection"]
        self.n_directions = 2 if self.bidirection else 1
        self.attention = config["attention"]
        self.input_size = 21
        self.embed_weight = F.one_hot(torch.arange(0, 21)).float()
        self.embed_size = self.embed_weight.shape[1]
        self.hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(
            self.input_size, self.embed_size, _weight=self.embed_weight)
        if self.model_type == "GRU":
            self.gru = nn.GRU(self.embed_size, self.hidden_size,
                              num_layers=3, dropout=self.dropout, bidirectional=self.bidirection)
        if self.model_type == "LSTM":
            self.lstm = nn.LSTM(self.embed_size, self.hidden_size,
                                num_layers=3, dropout=self.dropout, bidirectional=self.bidirection)
        self.atten = Attention(self.hidden_size * self.n_directions)
        self.linear = nn.Linear(self.hidden_size * self.n_directions, 1)

    def forward(self, x):
        x = x.t()
        embedding = self.embedding(x)
        # gru_inputs = pack_padded_sequence(embedding, seq_len)
        if self.model_type == "GRU":
            output, hidden = self.gru(embedding)
        if self.model_type == "LSTM":
            output, (hidden, _) = self.lstm(embedding)
        # outputs[seq_len, batch_size, hidden_size * n_directions]
        # hidden[num_layers * n_directions, batch_size, hidden_size]
        if self.n_directions == 2:
            hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
        else:
            hidden = hidden[-1]
        if self.attention:
            out = self.atten(output)
        else:
            out = hidden
        out = torch.sigmoid(self.linear(out))
        return out


def make_tensordataset(peptide):
    seq = [[aa_idx[aa] for aa in pep] for pep in peptide]
    seq_pad = np.zeros([len(peptide), 15])
    for i, p in enumerate(seq):
        for j, a in enumerate(p):
            seq_pad[i, j] = a
    seq_tensor = torch.LongTensor(seq_pad)
    return seq_tensor


def make_tensor_single_peptide(peptide):
    seq = [aa_idx[aa] for aa in peptide]
    seq_pad = np.zeros([1, 15])
    for i, a in enumerate(seq):
        seq_pad[0, i] = a
    seq_tensor = torch.LongTensor(seq_pad)
    return seq_tensor


def score2aff(pred_score):
    pred_score = np.array(pred_score)
    pred_aff = np.exp((1 - pred_score) * np.log(5e7))
    return pred_aff
