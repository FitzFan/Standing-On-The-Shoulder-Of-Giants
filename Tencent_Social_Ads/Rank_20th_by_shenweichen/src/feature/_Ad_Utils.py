#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Ad_Utils.py
@time: 2018/3/7 16:27
@desc: 数据预处理工具类
"""

import pickle
import pandas as pd
import numpy as np
import scipy.stats as sps  # 统计推断包

from tqdm import tqdm

# file_path
raw_data_path = 'E:/dataset/final/'
feature_data_path ='E:/dataset/final/features'
cache_pkl_path = '../cache_pkl/'
result_path = '../result/'


"""
- 初识pickle:
    - pickle是将数据持久化到磁盘的操作，为什么要pickle参见Somnus的回答：https://www.zhihu.com/question/38355589
    - 在实际操作中，更建议使用cPickle package来代替pickle，因为cPickle是C语言的版本实现，效率更高
    - Demo玩法参见：https://blog.oldj.net/2010/05/26/python-pickle/
- 玩转pickle:
    - 生成不同的feature后，使用pickle将其持久化到feature_path
"""


def load_pickle(path):
    return pickle.load(open(path, 'rb'))

def dump_pickle(obj, path, protocol=None):
    pickle.dump(obj, open(path, 'wb'), protocol=protocol)

# 正负例样本占比
def analyse(data, field):
    a = data.groupby(field).size()
    b = data.groupby(field)['label'].sum()
    c = pd.DataFrame({'conversion':b, 'click':a})
    c.reset_index(inplace=True)
    c['prob'] = c['conversion'] / c['click']
    return c.sort_values('prob', ascending=False)

def generate_file(valid_y, pred_prob):
    ans = valid_y.copy()
    ans['prob'] = pred_prob
    return ans

def addCrossFeature(split_train, split_test, feature_1, feature_2):
    '''
     根据训练集构造新的特征组合，对于测试集出现的新类别:取值NA
    :param split_train:
    :param split_test:
    :param feature_1:
    :param feature_2:
    :return:
    '''
    comb_index = split_train[[feature_1, feature_2]].drop_duplicates()
    comb_index[feature_1 + '_' + feature_2] = np.arange(1, comb_index.shape[0]+1) #在给定范围内给出给定范围内给定间隔的值
    split_train = pd.merge(split_train, comb_index, 'left', on=[feature_1, feature_2])
    split_test = pd.merge(split_test, comb_index, 'left', on=[feature_1, feature_2])
    return split_train, split_test

def get_feature_value(features, values, sort=True):
    '''
     获取特征和值
    :param features:
    :param values:
    :param sort:
    :return:
    '''
    df = pd.DataFrame({'name':features, 'value':values, 'abs_':np.abs(values)})
    if sort:
        return df.sort_values('abs_', ascending=False)
    else:
        return df

def feature_spearmanr(data, feature_list):
    '''
    :param data:
    :param feature_list:
    :return:
    '''
    cor_feature = []
    spearmanr = []
    for i in range(0, len(feature_list)):
        for j in range(i+1, len(feature_list)):
            cor_feature.append('_'.join([feature_list[i], feature_list[j]]))
            spearmanr.append(sps.spearmanr(data[feature_list[i]], data[feature_list[i]])[0])
    sp_df = pd.DataFrame({'feature':cor_feature, 'spearmanr':spearmanr})
    sp_df['abs_spearmanr'] = np.abs(sp_df['spearmanr'])
    sp_df.sort_values('abs_spearmanr', ascending=False, inplace=True)
    return sp_df

def feature_target_spearmanr(data, feature_list, target):
    cor_feature = []
    spearmanr = []
    for i in range(0, len(feature_list)):
        cor_feature.append('_'.join([feature_list, target]))
        spearmanr.append(sps.spearmanr(data[feature_list[i]], data[target])[0])
    sp_df = pd.DataFrame({'feature':cor_feature, 'spearmanr':spearmanr})
    sp_df['abs_spearmanr'] = np.abs(sp_df['spearmanr'])
    sp_df.sort_values('abs_spearmanr', ascending=False, inplace=True)
    return sp_df

def stratified_sampling(train, frac=0.33, seed=0):
    np.random.seed(seed)
    label_mean = train['label'].mean()
    pos_size = int(train.shape[0] * frac * label_mean)
    neg_size = int(train.shape[0] * frac * (1-label_mean))
    pos_index = train[train.label == 1].index
    neg_index = train[train.label == 0].index
    sample_pos_idx = np.random.choice(pos_index, pos_size, replace=False)  # 从给定的1-D数组生成随机样本
    sample_neg_idx = np.random.choice(neg_index, neg_size, replace=False)
    sample_index = np.hstack([sample_pos_idx, sample_neg_idx])
    np.random.shuffle(sample_index)
    return train.loc[sample_index]

def inverse_logit(x):
    return np.log(x/(1-x))

def logit(x):
    return 1/(1+np.exp(-x))

def calibration(pred, avg):
    intercept = inverse_logit(np.mean(pred)) - inverse_logit(avg)
    return logit(inverse_logit(pred) - intercept)

def simple_avg(pred_list):
    ans = np.ones_like(pred_list[0])
    for p in pred_list:
        ans += inverse_logit(p)
    return logit(ans/len(pred_list))
