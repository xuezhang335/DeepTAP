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
    seq_pad = np.zeros([len(peptide), 17])
    for i, p in enumerate(seq):
        for j, a in enumerate(p):
            seq_pad[i, j] = a
    seq_tensor = torch.LongTensor(seq_pad)
    return seq_tensor


def make_tensor_single_peptide(peptide):
    seq = [aa_idx[aa] for aa in peptide]
    seq_pad = np.zeros([1, 17])
    for i, a in enumerate(seq):
        seq_pad[0, i] = a
    seq_tensor = torch.LongTensor(seq_pad)
    return seq_tensor


def deeptap_main(args):
    CurDir = os.path.dirname(os.path.realpath(__file__))
    i = datetime.datetime.now()
    print(f"{i}: Prediction starting ... \n")
    if args.file:
        test_file = pd.read_csv(args.file)
        test_peptide = test_file["peptide"]
        test_data = make_tensordataset(test_peptide)
    else:
        test_peptide = args.peptide
        test_data = make_tensor_single_peptide(test_peptide)
    if args.outputDir:
        Dir = args.outputDir
    else:
        Dir = "."
    predScores = torch.zeros(5, len(test_data))
    for i in range(5):
        path = f"{CurDir}/model/{i+1}.ckpt"
        checkpoint = torch.load(path)
        config = checkpoint["hyper_parameters"]
        model = Model.load_from_checkpoint(path, config=config)  # 加载了参数
        model.eval()
        model.freeze()
        y_hat = model(test_data)
        predScores[i] = torch.squeeze(y_hat)
    pred_score = predScores.mean(dim=0)
    pred_label = [1 if i >= 0.5 else 0 for i in pred_score]
    date = datetime.datetime.now()
    date = f"{date.year}_{date.month}_{date.day}"
    with open(f"{Dir}/{date}_TAPPred_prediction_result.csv", "w")as f:
        f.write("peptide,pred_score,pred_label\n")
        if args.file:
            for i, _ in enumerate(test_peptide):
                f.write(
                    f"{test_peptide[i]},{pred_score[i]:.4f},{pred_label[i]}\n")
        else:
            f.write(f"{test_peptide},{pred_score[0]:.4f}\n")
    j = datetime.datetime.now()
    print(f"{j}: Prediction end.\n")
