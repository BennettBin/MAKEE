# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project: makee 
# @File: makee_13P_TPE.py
# @Date: 2025/7/21 11:19
# @Author: binb_chen@163.com
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import csv
import os
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

"""参数设置"""

DATASET = "BPIC20_D"
ATTRIBUTES = ["concept:name", "org:resource", "org:role"]  # 第一个必须为activity name
# ["concept:name", "org:resource", "org:role"]

GPU_ID = 3
RANDOM_SEED = 0
MAX_EPOCH = 100
EMBEDDING_DIM = 32
FOLD = 3
TRIALS = 20

# # 固定参数
# BATCH = 32
# DROPOUT_RATE = 0.5  # 0.1-0.9
# HIDDEN_DIM = 128  # 64, 128, 256, 512
# LEARN_RATE = 0.0001

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.set_device(GPU_ID)  # 指定GPU
    DEVICE = 'cuda:' + str(GPU_ID)


"""数据处理"""


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def max_min(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())


def normalise(dataframe):
    return (dataframe - dataframe.mean()) / (dataframe.std() + 1e-10)


def read_log(log_add_, case_num_):
    log_list = [[] for i in range(case_num_)]
    with open(log_add_, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            log_list[int(row['case']) - 1].append(row)
    f.close()

    return log_list


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


"""早停法"""


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='resAttention.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


"""数据处理"""


class XesEventLog:
    def __init__(self, chosen_attributes, min_prefix_length=2):

        self.log_add = "dataset/" + DATASET + ".csv"
        self.log_df = pd.read_csv(self.log_add)
        self.case_num = self.log_df.iloc[-1].iloc[0]
        self.event_attributes = self.log_df.keys().to_list()
        self.attributes_values, self.attributes_values_nums = dict(), dict()
        for attribute in self.event_attributes:
            self.attributes_values[attribute] = list(set(self.log_df[attribute].values.tolist()))
            self.attributes_values_nums[attribute] = len(self.attributes_values[attribute])

        self.min_prefix_length = min_prefix_length
        self.log_list = read_log(self.log_add, self.case_num)

        self.max_length = max([len(trace) for trace in self.log_list])  # 最长的trace长度, 用来确定事件级LSTM的输入维度
        self.chosen_attributes = chosen_attributes
        self.case_id_key = "case:concept:name"
        if "concept:name" in self.event_attributes:  # 活动所对应的的列名, 一般为concept:name
            self.activity_key = "concept:name"
            self.timestamp_key = "time:timestamp"
        else:
            self.activity_key = "Activity"
            self.timestamp_key = "timestamp"
        self.max_atts = max([att_num for att_num in self.attributes_values_nums.values()])
        self.encoding_length = EMBEDDING_DIM  # 属性编码的长度
        self.activities = self.attributes_values[self.activity_key]  # 固定顺序的活动列表
        self.activity_labels = {self.activities[i]: i for i in range(len(self.activities))}

        self.attribute_encodings = dict()
        for attribute in self.chosen_attributes:
            if attribute in ['Complete Timestamp', 'time:timestamp', 'ComplaintThemeID', 'dateFinished', 'timestamp']:
                self.attribute_encodings[attribute] = self.time_encoding(attribute)  # 其它属性编码结果
            else:
                self.attribute_encodings[attribute] = self.embedding(attribute)

    def dfg_get(self, attribute):
        dfg_dict = dict()
        for trace in self.log_list:
            attribute_list = [event[attribute] for event in trace]
            for i in range(len(attribute_list) - 1):
                key = (attribute_list[i], attribute_list[i + 1])
                if key not in dfg_dict.keys():
                    dfg_dict[key] = 1
                else:
                    dfg_dict[key] += 1
        return dfg_dict

    def time_encoding(self, target_attribute):
        encoding_result = dict()
        for i in range(len(self.attributes_values[target_attribute])):
            encoder_t = [0 for i in
                         range(self.encoding_length - len(str(self.attributes_values[target_attribute][i])))]
            for s in str(self.attributes_values[target_attribute][i]):
                encoder_t.append(int(s))
            if max(encoder_t) != 0:
                new_encoder_t = []
                for s in encoder_t:
                    new_encoder_t.append(round(s / max(encoder_t), 4))
                encoding_result[self.attributes_values[target_attribute][i]] = new_encoder_t
            else:
                encoding_result[self.attributes_values[target_attribute][i]] = encoder_t

        return encoding_result

    def embedding(self, attribute):
        attribute_num = self.attributes_values_nums[attribute]
        num_list = torch.IntTensor([i for i in range(attribute_num)])
        embedding = nn.Embedding(attribute_num, self.encoding_length)
        results = embedding(num_list)
        embedding_result = dict()
        for j in range(attribute_num):
            embedding_result[self.attributes_values[attribute][j]] = results[j].tolist()
        return embedding_result

    def fit_prefix(self, prefix_):
        """
        给prefix补零，使长度统一
        :param prefix_: list, 一个待补零的prefix
        :return: list, 已经补充完整的prefix
        """
        unfitted_prefix = []
        fit_list = [[0 for i in range(self.encoding_length)] for j in range(len(prefix_[0]))]
        if len(prefix_) < self.max_length - 1:
            for i in range(self.max_length - len(prefix_) - 1):
                unfitted_prefix.append(fit_list)
        new_prefix = unfitted_prefix + prefix_  # 前序补零

        return new_prefix

    def prefix_encoding(self, prefix_dict):
        prefix_encoding = []
        for i in range(len(prefix_dict)):
            event_encoding = []
            for attribute in self.chosen_attributes:
                if prefix_dict[i][attribute] in self.attributes_values[attribute]:
                    event_encoding.append(self.attribute_encodings[attribute][prefix_dict[i][attribute]])
                else:
                    event_encoding.append([0 for i in range(self.encoding_length)])
            prefix_encoding.append(event_encoding)
        prefix_encoding = self.fit_prefix(prefix_encoding)

        return prefix_encoding

    def dfg_to_matrix(self, attribute, dfg):
        """

        Returns
        -------
            matrix_relation: List, 活动之间的关联矩阵; 有向, 方向为从行名到列名
            matrix_weight: List, 活动之间的权重矩阵; 有向, 方向为从行名到列名
        """
        matrix_relation_ = pd.DataFrame([[0 for i in range(self.attributes_values_nums[attribute])] for j in
                                         range(self.attributes_values_nums[attribute])],
                                        index=self.attributes_values[attribute],
                                        columns=self.attributes_values[attribute])
        matrix_weight_ = pd.DataFrame([[0 for i in range(self.attributes_values_nums[attribute])] for j in
                                       range(self.attributes_values_nums[attribute])],
                                      index=self.attributes_values[attribute],
                                      columns=self.attributes_values[attribute])

        for key, val in dfg.items():
            matrix_relation_.loc[key[0], key[1]] += 1
            matrix_weight_.loc[key[0], key[1]] += val

        return matrix_relation_.values.tolist(), matrix_weight_.values.tolist()

    def log_to_matrices(self, attribute):
        """

        Returns
        -------
            np.array(prefixes): List[List]
            np.array(next_activities): Array[Str]
        """
        print(f"-------------{attribute}-------------\n")
        prefix_weights_ = []
        for trace in tqdm(self.log_list):
            trace_df = pd.DataFrame(trace)
            for i in range(len(trace_df) - self.min_prefix_length):
                prefix_ = trace_df[:i + self.min_prefix_length]
                matrix_weight_ = np.array([[0 for i in range(self.attributes_values_nums[attribute])] for j in
                                           range(self.attributes_values_nums[attribute])])
                for j in range(len(prefix_[attribute].values) - 1):
                    matrix_weight_[self.attributes_values[attribute].index(prefix_[attribute].iloc[j])][
                        self.attributes_values[attribute].index(prefix_[attribute].iloc[j + 1])] += 1

                prefix_weights_.append(normalise(pd.DataFrame(matrix_weight_)).values.tolist())

        return np.array(prefix_weights_)

    def log_to_prefix_encodings(self):
        prefixes_encodings_, next_activities_ = [], []
        for trace in tqdm(self.log_list):
            trace_df = pd.DataFrame(trace)
            for i in range(len(trace_df) - self.min_prefix_length):
                prefix_ = trace_df[:i + self.min_prefix_length]
                prefixes_dict = prefix_.to_dict('records')  # 转化成list[dict]格式, 每一个dict代表一个事件
                prefix_encoding = self.prefix_encoding(prefixes_dict)
                prefixes_encodings_.append(prefix_encoding)
                next_activities_.append(trace_df[self.activity_key].iloc[i + self.min_prefix_length])

        labels = [self.activity_labels[activity] for activity in next_activities_]

        return np.array(prefixes_encodings_), np.array(labels)


def data_pro(chosen_attributes=None):  # chosen_attributes=["Activity"]

    if chosen_attributes is None:
        chosen_attributes = ["Activity"]

    event_log = XesEventLog(chosen_attributes)  # 目标事件日志

    whole_data_add = "whole_data/" + DATASET
    mkdir(whole_data_add)  # 调用函数

    event_log_parameters_save_add = whole_data_add + "/" + DATASET + "_log_parameters.csv"
    f = open(event_log_parameters_save_add, 'a', encoding='utf-8')
    open(event_log_parameters_save_add, "r+").truncate()
    f.write("chosen_attribute_num,encoding_length,max_prefix_length\n")
    f.write(f"{len(event_log.chosen_attributes)},{event_log.encoding_length},{event_log.max_length - 1}\n")
    f.close()

    input_dims_save_add = whole_data_add + "/" + DATASET + "_attributes_values_nums.csv"
    f = open(input_dims_save_add, 'a', encoding='utf-8')
    open(input_dims_save_add, "r+").truncate()
    for k, v in event_log.attributes_values_nums.items():
        f.write(f"{k},{v}\n")
    f.close()

    prefix_encodings, labels = event_log.log_to_prefix_encodings()  # tqdm
    prefix_encodings_add = whole_data_add + "/" + DATASET + "_prefix_encodings.npy"
    np.save(prefix_encodings_add, np.array(prefix_encodings))
    labels_add = whole_data_add + "/" + DATASET + "_labels.npy"
    np.save(labels_add, labels)

    for attribute in chosen_attributes:
        dfg = event_log.dfg_get(attribute)
        dfg_relations, dfg_weights = event_log.dfg_to_matrix(attribute, dfg)
        prefix_weights = event_log.log_to_matrices(attribute)  # tqdm

        dfg_relation_add = whole_data_add + "/" + DATASET + "_" + attribute + "_dfg_relation.npy"
        dfg_weight_add = whole_data_add + "/" + DATASET + "_" + attribute + "_dfg_weight.npy"
        prefix_weights_add = whole_data_add + "/" + DATASET + "_" + attribute + "_prefix_weights.npy"

        np.save(dfg_relation_add, np.array(dfg_relations))
        np.save(dfg_weight_add, np.array(dfg_weights))
        np.save(prefix_weights_add, np.array(prefix_weights))

    """分割数据"""
    print("--------------分割数据--------------")

    dfg_relations, dfg_weights = dict(), dict()
    for attribute in ATTRIBUTES:
        dfg_relation_add = "whole_data/" + DATASET + "/" + DATASET + "_" + attribute + "_dfg_relation.npy"
        dfg_weight_add = "whole_data/" + DATASET + "/" + DATASET + "_" + attribute + "_dfg_weight.npy"
        dfg_relations[attribute] = np.load(dfg_relation_add)
        dfg_weights[attribute] = np.load(dfg_weight_add)

    under_split_arrays = []  # 依次是若干个属性的prefix_weights
    for attribute in ATTRIBUTES:
        prefix_weights_add = "whole_data/" + DATASET + "/" + DATASET + "_" + attribute + "_prefix_weights.npy"
        prefix_weights = np.load(prefix_weights_add)
        under_split_arrays.append(prefix_weights)

    prefix_encodings_add = "whole_data/" + DATASET + "/" + DATASET + "_prefix_encodings.npy"
    labels_add = "whole_data/" + DATASET + "/" + DATASET + "_labels.npy"
    prefix_encodings = np.load(prefix_encodings_add)
    labels = np.load(labels_add)
    under_split_arrays.append(prefix_encodings)
    under_split_arrays.append(labels)

    all_tuple = train_test_split(*under_split_arrays, test_size=0.2)
    back_all_tuple = all_tuple[-4:]  # 分别是prefix_encodings和labels
    new_all_tuple = all_tuple[:-4]  # prefix_weights

    """保存测试数据"""
    test_data_add = "test_data/" + DATASET + "/test.csv/"
    mkdir(test_data_add)  # 调用函数
    # 测试集
    test_prefix_encodings = back_all_tuple[1]
    test_labels = back_all_tuple[3]
    test_prefix_encodings_save_add = test_data_add + DATASET + "_test_prefix_encodings.npy"
    np.save(test_prefix_encodings_save_add, test_prefix_encodings)
    test_label_save_add = test_data_add + DATASET + "_test_labels.npy"
    np.save(test_label_save_add, test_labels)

    test_prefix_weights = dict()
    for i in range(len(new_all_tuple)):
        for j in range(len(ATTRIBUTES)):
            if i % 2 == 1 and math.floor(i / 2) == j:
                test_prefix_weights[ATTRIBUTES[j]] = new_all_tuple[i]
                test_save_add = test_data_add + DATASET + "_test_prefix_weights_" + ATTRIBUTES[j] + ".npy"
                np.save(test_save_add, new_all_tuple[i])

    # 训练集与验证集

    kf = KFold(n_splits=FOLD)
    for k, (fold_train_index, fold_valid_index) in enumerate(kf.split(new_all_tuple[0])):
        fold_data_add = "fold_data/" + DATASET + "/fold" + str(k + 1) + "/"
        mkdir(fold_data_add)  # 调用函数
        # 第k个fold
        train_prefix_encodings_fold = np.array([back_all_tuple[0][idx] for idx in fold_train_index])
        valid_prefix_encodings_fold = np.array([back_all_tuple[0][idx] for idx in fold_valid_index])
        train_labels_fold = np.array([back_all_tuple[2][idx] for idx in fold_train_index])
        valid_labels_fold = np.array([back_all_tuple[2][idx] for idx in fold_valid_index])

        train_prefix_encodings_save_add = fold_data_add + DATASET + "_train_prefix_encodings_fold" + str(
            k + 1) + ".npy"
        valid_prefix_encodings_save_add = fold_data_add + DATASET + "_valid_prefix_encodings_fold" + str(
            k + 1) + ".npy"
        train_labels_save_add = fold_data_add + DATASET + "_train_labels_fold" + str(k + 1) + ".npy"
        valid_labels_save_add = fold_data_add + DATASET + "_valid_labels_fold" + str(k + 1) + ".npy"

        np.save(train_prefix_encodings_save_add, train_prefix_encodings_fold)
        np.save(valid_prefix_encodings_save_add, valid_prefix_encodings_fold)
        np.save(train_labels_save_add, train_labels_fold)
        np.save(valid_labels_save_add, valid_labels_fold)

        train_prefix_weights, valid_prefix_weights = dict(), dict()
        for i in range(len(new_all_tuple)):
            for j in range(len(ATTRIBUTES)):
                if i % 2 == 0 and math.floor(i / 2) == j:
                    train_prefix_weights_fold = np.array([new_all_tuple[i][idx] for idx in fold_train_index])
                    valid_prefix_weights_fold = np.array([new_all_tuple[i][idx] for idx in fold_valid_index])
                    train_prefix_weights[ATTRIBUTES[j]] = train_prefix_weights_fold
                    valid_prefix_weights[ATTRIBUTES[j]] = valid_prefix_weights_fold

                    train_prefix_weights_fold_save_add = fold_data_add + DATASET + "_train_prefix_weights_" + \
                                                         ATTRIBUTES[j] + "_fold" + str(k + 1) + ".npy"
                    valid_prefix_weights_fold_save_add = fold_data_add + DATASET + "_valid_prefix_weights_" + \
                                                         ATTRIBUTES[j] + "_fold" + str(k + 1) + ".npy"
                    np.save(train_prefix_weights_fold_save_add, train_prefix_weights_fold)
                    np.save(valid_prefix_weights_fold_save_add, valid_prefix_weights_fold)


"""模型构建"""


class GraphConv(torch.nn.Module):
    def __init__(self, num_features, out_channels):
        super(GraphConv, self).__init__()

        self.in_channels = num_features
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(num_features, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        x = adj @ x @ self.weight
        x = x + self.bias
        return x


class GCN(torch.nn.Module):
    def __init__(self, attribute_val_num_, input_dim_, hidden_dim, dfg_relation_):
        super().__init__()
        self.attribute_val_num = attribute_val_num_  # 活动数量
        self.input_dim = input_dim_  # 输入维度, 直接使用关联矩阵的时候等于活动数量
        self.hidden_dim = hidden_dim  # 隐层维度, 需要调参获取最优值
        self.dfg_relation = dfg_relation_
        self.conv1 = GraphConv(self.input_dim, self.hidden_dim)
        self.conv2 = GraphConv(self.hidden_dim, self.hidden_dim)
        self.weight1 = nn.Parameter(torch.ones((self.attribute_val_num, self.attribute_val_num), requires_grad=True))
        self.bias = nn.Parameter(torch.ones(self.attribute_val_num, requires_grad=True))
        self.weight2 = nn.Parameter(torch.ones((self.attribute_val_num, self.attribute_val_num), requires_grad=True))
        self.bias2 = nn.Parameter(torch.ones((self.attribute_val_num, self.attribute_val_num), requires_grad=True))
        self.flatten = nn.Flatten(0, -1)
        self.linear = nn.Linear(self.attribute_val_num * self.hidden_dim, self.hidden_dim)

    def forward(self, prefix_weight):
        """

        Parameters
        ----------
        Returns
        -------

        """
        prefix_relation = torch.tensor(np.where(prefix_weight.cpu() != 0, 1, prefix_weight.cpu()))
        batch_size = len(prefix_weight)
        if USE_CUDA:
            x = prefix_weight.cuda()
            prefix_relation = prefix_relation.cuda()
            dfg_rel = torch.tensor(self.dfg_relation, dtype=torch.float).cuda()
        else:
            x = prefix_weight.cpu()
            prefix_relation = prefix_relation.cpu()
            dfg_rel = torch.tensor(self.dfg_relation, dtype=torch.float).cpu()

        dfg_relation = dfg_rel.unsqueeze(0)
        dfg_relation = dfg_relation.repeat(batch_size, 1, 1)
        feat = x @ self.weight1 + dfg_relation @ self.weight2 + self.bias  # [32, 14, 14]
        y = F.leaky_relu(self.conv1(feat, prefix_relation))
        out = F.leaky_relu(self.conv2(y, prefix_relation))
        out = self.flatten(out).view(batch_size, -1)
        out = self.linear(out)
        return out


class Embedding(torch.nn.Module):
    def __init__(self, hidden_dim, encoding_length, num_layers=6, nhead=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoding_length = encoding_length
        self.nhead = nhead
        self.num_layers = num_layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.weight = nn.Parameter(torch.ones((self.encoding_length, self.hidden_dim), requires_grad=True))
        self.bias = nn.Parameter(torch.ones(self.hidden_dim, requires_grad=True))

    def forward(self, x):
        y = self.transformer_encoder(x @ self.weight + self.bias)
        return y


class XGCN(torch.nn.Module):
    def __init__(self, dfg_relations_, config_):
        super().__init__()
        self.hidden_dim = config_["hidden_dim"]  # 隐层维度, 需要调参获取最优值
        self.encoding_length = config_["log_parameters"]["encoding_length"]
        self.max_prefix_length = config_["log_parameters"]["max_prefix_length"]
        self.chosen_attribute_num = config_["log_parameters"]["chosen_attribute_num"]
        self.activity_num = config_["attributes_values_nums"][ATTRIBUTES[0]]
        self.dropout_rate = config_['dropout_rate']

        self.attribute_val_num1 = config_["attributes_values_nums"][ATTRIBUTES[0]]
        self.input_dim1 = config_["attributes_values_nums"][ATTRIBUTES[0]]
        self.dfg_relation1 = dfg_relations_[ATTRIBUTES[0]]

        self.attribute_val_num2 = config_["attributes_values_nums"][ATTRIBUTES[1]]
        self.input_dim2 = config_["attributes_values_nums"][ATTRIBUTES[1]]
        self.dfg_relation2 = dfg_relations_[ATTRIBUTES[1]]

        self.attribute_val_num3 = config_["attributes_values_nums"][ATTRIBUTES[2]]
        self.input_dim3 = config_["attributes_values_nums"][ATTRIBUTES[2]]
        self.dfg_relation3 = dfg_relations_[ATTRIBUTES[2]]

        self.gcn1 = GCN(self.attribute_val_num1, self.input_dim1, self.hidden_dim, self.dfg_relation1)
        self.gcn2 = GCN(self.attribute_val_num2, self.input_dim2, self.hidden_dim, self.dfg_relation2)
        self.gcn3 = GCN(self.attribute_val_num3, self.input_dim3, self.hidden_dim, self.dfg_relation3)

        self.embed1 = Embedding(self.hidden_dim, self.encoding_length)
        self.embed2 = Embedding(self.hidden_dim, self.encoding_length)
        self.embed3 = Embedding(self.hidden_dim, self.encoding_length)
        self.linear1 = nn.Linear(len(ATTRIBUTES) * self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Linear(len(ATTRIBUTES) * self.max_prefix_length * self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(2 * self.hidden_dim, self.activity_num)  # (len(ATTRIBUTES) + 1) *
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, tuple_):
        batch_size = len(tuple_[0])
        x1 = self.gcn1(tuple_[2])  # [32, 128]
        x2 = self.gcn2(tuple_[3])
        x3 = self.gcn3(tuple_[4])

        x = torch.concat([x1, x2, x3], dim=1)
        x = F.leaky_relu(self.linear1(x))  # 32, 128

        prefix_encoding = tuple_[0].permute(2, 1, 0, 3)
        coding1 = self.embed1(prefix_encoding[0].permute(1, 0, 2)).view(batch_size, -1)  # 32, 14, 128
        coding2 = self.embed2(prefix_encoding[1].permute(1, 0, 2)).view(batch_size, -1)
        coding3 = self.embed3(prefix_encoding[2].permute(1, 0, 2)).view(batch_size, -1)
        z = torch.concat([coding1, coding2, coding3], dim=1)
        z = F.leaky_relu(self.linear2(z))  # 32, 128
        y = torch.concat([x, z], dim=1)
        y = self.dropout(y)
        out = self.linear(y)

        return out


"""模型训练"""


def run_test(net, test_prefix_weights_, test_prefix_encodings_, test_labels_, test_result_add):
    integrated_test = []
    integrated_test.append(torch.tensor(test_prefix_encodings_, dtype=torch.float))
    integrated_test.append(torch.tensor(test_labels_, dtype=torch.float))
    for attribute in ATTRIBUTES:
        integrated_test.append(torch.tensor(test_prefix_weights_[attribute], dtype=torch.float))

    test_datas = tuple(integrated_test)
    test_data = TensorDataset(*test_datas)

    attributes_values_nums_add = "whole_data/" + DATASET + "/" + DATASET + "_attributes_values_nums.csv"
    attributes_values_nums = dict()
    f = open(attributes_values_nums_add, encoding='utf-8')
    for row in csv.reader(f):
        attributes_values_nums[row[0]] = int(row[1])
    f.close()

    if USE_CUDA:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(test_data, shuffle=True, drop_last=True)
    net.eval()
    all_labels_test, all_predict_test = [], []
    test_losses = []
    predict_probs = []
    with torch.no_grad():  # 表示不需要保存训练参数过程中的梯度
        for i, tuple_test in enumerate(test_loader):
            if USE_CUDA:
                list_test = []
                for k in range(len(ATTRIBUTES) + 2):
                    list_test.append(tuple_test[k].cuda())
                tuple_test = tuple(list_test)
            output_test = net(tuple_test)
            test_loss = criterion(output_test, tuple_test[1].long())
            test_losses.append(test_loss.item())
            # 计算准确度
            predict_output = torch.nn.Softmax(dim=1)(output_test)
            predict_ = torch.argmax(predict_output, 1)

            predict_probs += predict_output.tolist()
            all_labels_test += tuple_test[1].tolist()
            all_predict_test += predict_.tolist()

    test_loss = np.average(test_losses)
    test_acc = metrics.accuracy_score(all_labels_test, all_predict_test)
    test_macro_f = metrics.f1_score(all_labels_test, all_predict_test, average='macro', zero_division=0)

    with open(test_result_add, 'a', encoding='utf-8') as f:
        f.write(
            f'{test_loss},{test_acc},{test_macro_f}\n')
    f.close()

    print(f'results ===>>> '
          f'test_loss: {test_loss} | test_acc: {test_acc} | test_macro_f: {test_macro_f}\n')
    return test_acc


def objective(trial):
    # 超参数设置
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.001, log=False)
    dropout_rate = trial.suggest_float("drop_rate", 0.1, 0.9, log=False)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step=64)
    # max_features = trial.suggest_categorical("max_features",["log2","sqrt","auto"]) #字符型

    # 测试结果设置
    mkdir("results/")
    test_write_add = f"results/{os.path.basename(__file__)[:-3]}_result.csv"
    with open(test_write_add, 'a', encoding='utf-8') as f:
        f.write(f'----------{RANDOM_SEED}----------\n')
        f.write(
            f'----batch_size={batch_size}, learning_rate={learning_rate}, dropout_rate={dropout_rate}, hidden_dim={hidden_dim}----\n')
        f.write('test_loss,test_acc,test_macro_f\n')
    f.close()

    # 读取dfg数据
    dfg_relations, dfg_weights = dict(), dict()
    for attribute in ATTRIBUTES:
        dfg_relation_add = "whole_data/" + DATASET + "/" + DATASET + "_" + attribute + "_dfg_relation.npy"
        dfg_weight_add = "whole_data/" + DATASET + "/" + DATASET + "_" + attribute + "_dfg_weight.npy"
        dfg_relations[attribute] = np.load(dfg_relation_add)
        dfg_weights[attribute] = np.load(dfg_weight_add)

    avg_valid_losses = []
    # 训练与测试
    for k in range(FOLD):

        print(f"-----------------------训练模型: fold {k + 1}-----------------------")
        # 训练数据与验证数据读取
        fold_data_add = "fold_data/" + DATASET + "/fold" + str(k + 1) + "/"
        mkdir(fold_data_add)  # 调用函数
        # 第k个fold
        train_prefix_encodings_fold_add = "fold_data/" + DATASET + "/fold" + str(
            k + 1) + "/" + DATASET + "_train_prefix_encodings_fold" + str(k + 1) + ".npy"
        valid_prefix_encodings_fold_add = "fold_data/" + DATASET + "/fold" + str(
            k + 1) + "/" + DATASET + "_valid_prefix_encodings_fold" + str(k + 1) + ".npy"
        train_labels_fold_add = "fold_data/" + DATASET + "/fold" + str(
            k + 1) + "/" + DATASET + "_train_labels_fold" + str(k + 1) + ".npy"
        valid_labels_fold_add = "fold_data/" + DATASET + "/fold" + str(
            k + 1) + "/" + DATASET + "_valid_labels_fold" + str(k + 1) + ".npy"
        train_prefix_encodings_fold = np.load(train_prefix_encodings_fold_add)
        valid_prefix_encodings_fold = np.load(valid_prefix_encodings_fold_add)
        train_labels_fold = np.load(train_labels_fold_add)
        valid_labels_fold = np.load(valid_labels_fold_add)

        train_prefix_weights, valid_prefix_weights = dict(), dict()
        for attribute in ATTRIBUTES:
            prefix_weights_train_fold_add = ("fold_data/" + DATASET + "/fold" + str(
                k + 1) + "/" + DATASET + "_train_prefix_weights_" + attribute + "_fold" + str(k + 1) + ".npy")
            prefix_weights_valid_fold_add = ("fold_data/" + DATASET + "/fold" + str(
                k + 1) + "/" + DATASET + "_valid_prefix_weights_" + attribute + "_fold" + str(k + 1) + ".npy")
            train_prefix_weights[attribute] = np.load(prefix_weights_train_fold_add)
            valid_prefix_weights[attribute] = np.load(prefix_weights_valid_fold_add)

        model_path_fold = "results/" + os.path.basename(__file__)[:-3] + "_model_" + str(k + 1) + ".pth"
        log_parameters_add = "whole_data/" + DATASET + "/" + DATASET + "_log_parameters.csv"
        log_parameters = dict()
        f = open(log_parameters_add, encoding='utf-8')
        for row in csv.DictReader(f):
            for k, v in row.items():
                log_parameters[k] = int(v)
        f.close()

        attributes_values_nums_add = "whole_data/" + DATASET + "/" + DATASET + "_attributes_values_nums.csv"
        attributes_values_nums = dict()
        f = open(attributes_values_nums_add, encoding='utf-8')
        for row in csv.reader(f):
            attributes_values_nums[row[0]] = int(row[1])
        f.close()

        config = {
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "log_parameters": log_parameters,
            "attributes_values_nums": attributes_values_nums
        }

        integrated_train, integrated_valid = [], []
        integrated_train.append(torch.tensor(train_prefix_encodings_fold, dtype=torch.float))
        integrated_train.append(torch.tensor(train_labels_fold, dtype=torch.float))
        for attribute in ATTRIBUTES:
            integrated_train.append(torch.tensor(train_prefix_weights[attribute], dtype=torch.float))

        integrated_valid.append(torch.tensor(valid_prefix_encodings_fold, dtype=torch.float))
        integrated_valid.append(torch.tensor(valid_labels_fold, dtype=torch.float))
        for attribute in ATTRIBUTES:
            integrated_valid.append(torch.tensor(valid_prefix_weights[attribute], dtype=torch.float))

        train_datas = tuple(integrated_train)
        valid_datas = tuple(integrated_valid)
        train_data = TensorDataset(*train_datas)
        valid_data = TensorDataset(*valid_datas)

        net = XGCN(dfg_relations, config)
        if USE_CUDA:
            net.cuda()

        print(f'模型参数为{sum(x.numel() for x in net.parameters())}')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True, drop_last=True)

        # net.cuda()
        all_labels_train, all_predict_train = [], []
        early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path_fold)
        train_losses, valid_losses = [], []
        for epoch in range(MAX_EPOCH):
            net.train()
            for i, tuple_train in enumerate(train_loader):
                if USE_CUDA:
                    list_train = []
                    for k in range(len(ATTRIBUTES) + 2):
                        list_train.append(tuple_train[k].cuda())
                    tuple_train = tuple(list_train)
                    # tuple_train = (tuple_train[0].cuda(), tuple_train[1].cuda(), tuple_train[2].cuda())
                optimizer.zero_grad()
                output_train = net(tuple_train)
                train_loss = criterion(output_train, tuple_train[1].long())
                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.item())

                # 计算准确度
                predict = torch.nn.Softmax(dim=1)(output_train)
                predict = torch.argmax(predict, 1)
                all_labels_train += tuple_train[1].cpu().numpy().tolist()
                all_predict_train += predict.cpu().numpy().tolist()

            # 检查是否值得继续训练，不值得将会中止
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # 验证
            net.eval()
            all_labels_valid, all_predict_valid, pred_probs = [], [], []
            with torch.no_grad():  # 表示不需要保存训练参数过程中的梯度
                for j, tuple_valid in enumerate(valid_loader):
                    if USE_CUDA:
                        list_valid = []
                        for k in range(len(ATTRIBUTES) + 2):
                            list_valid.append(tuple_valid[k].cuda())
                        tuple_valid = tuple(list_valid)
                    output_valid = net(tuple_valid)
                    val_loss = criterion(output_valid, tuple_valid[1].long())
                    # scheduler.step(val_loss)
                    valid_losses.append(val_loss.item())
                    # 计算准确度
                    predict_output = torch.nn.Softmax(dim=1)(output_valid)
                    predict_test = torch.argmax(predict_output, 1)
                    all_labels_valid += tuple_valid[1].tolist()
                    all_predict_valid += predict_test.tolist()
                    pred_probs += predict_output.tolist()

            avg_train_loss_epoch = np.average(train_losses)
            avg_valid_loss_epoch = np.average(valid_losses)
            avg_valid_losses.append(avg_valid_loss_epoch)
            train_acc = metrics.accuracy_score(all_labels_train, all_predict_train)
            valid_acc = metrics.accuracy_score(all_labels_valid, all_predict_valid)

            print("| Epoch: {:2d}/{:2d} |".format(epoch + 1, MAX_EPOCH),
                  "Train Loss: {:.4f} |".format(avg_train_loss_epoch),
                  "Valid Loss: {:.4f} |".format(avg_valid_loss_epoch),
                  "Train Acc: {:.4f} |".format(train_acc),
                  "Valid Acc: {:.4f} |".format(valid_acc),
                  )
            # 清空loss列表
            train_losses = []
            valid_losses = []
            torch.save(net, model_path_fold)

            trial.report(avg_valid_loss_epoch, epoch)

            early_stopping(avg_valid_loss_epoch, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print(f"-----------------------测试模型-----------------------")
        test_prefix_weights = dict()
        for attribute in ATTRIBUTES:
            test_prefix_weights_add = (
                    "test_data/" + DATASET + "/test.csv/" + DATASET + "_test_prefix_weights_" + attribute + ".npy")
            test_prefix_weights[attribute] = np.load(test_prefix_weights_add)

        test_prefix_encodings = np.load("test_data/" + DATASET + "/test.csv/" + DATASET + "_test_prefix_encodings.npy")
        test_labels = np.load("test_data/" + DATASET + "/test.csv/" + DATASET + "_test_labels.npy")

        run_test(net, test_prefix_weights, test_prefix_encodings, test_labels, test_write_add)

    return sum(avg_valid_losses) / len(avg_valid_losses)


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_trail_parameters_save_add = f"results/{os.path.basename(__file__)[:-3]}_best_parameters.csv"
    with open(best_trail_parameters_save_add, 'a', encoding='utf-8') as f:
        f.write(f'Parameters of the best trial:\n')
        for key, value in trial.params.items():
            f.write(f'{key}: {value}\n')
    f.close()


if __name__ == '__main__':
    # 数据处理
    data_pro(chosen_attributes=ATTRIBUTES)
    # 模型训练、调参与测试
    main()
