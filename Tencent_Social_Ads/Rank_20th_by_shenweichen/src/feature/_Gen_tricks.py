#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Gen_tricks.py
@time: 2018/3/14 9:03
@desc: 广告主转化回流上报机制分析（根据advertiserID）
       不同的广告主具有不同的转化计算方式，如第一次点击算转化，最后一次点击算转化，安装时点击算转化，分析并构造相应描述特征，提升模型预测精度。
       PS：如果业务背景不是转化机制，而是电商购买。add_diff()的作用是：
        - 统计不同首次点击和当前点击的时间差，末次点击和当前点击的时间差。
        - 其实应该加入天级统计，这样特征的粒度更细，模型能学到的东西更多。
        - 参见bettenW的102_trick_feat.py的代码。
"""
import pandas as pd
import numpy as np
import gc
import os

from Smooth import BayesianSmoothing
from tqdm import tqdm
from Ad_Utils import raw_data_path, feature_data_path, load_pickle, dump_pickle
from Feature_joint import addAd, addPosition, addTime


def trick(row):
    if row['ua_cnt'] <= 1:  # 如果ua_cnt=1 <=> 一次点击就转化了 <=> 不存在第一次点击or最后一点击or安装时的点击之分。
        return 0
    elif row['ua_first'] > 0:
        return 1
    elif row['ua_last'] > 0:
        return 2
    else:
        return 3

def add_trick(df):
    """
    ** 新学trick： 判断不同广告主衡量转化的方式
    - 根据ua_id和adv_id进行聚合，找出发生转化的组合，并计算转化数量；
    - 当有多次点击时，先根据user_id, adv_id, clk_time进行复合排序；
    - 根据去重操作的keep参数的不同，构造不同的data_frame；
    - 对每一行施加trick()函数。（复习一下：apply(trick, axis=1), axis=1 <=>apply function to each row)
    - 最后的trick列，表示转化的方式
    - 待优化之处：
        - freelzy大牛考虑了更高阶的trick数据： 'creativeID', 'positionID', 'adID', 'appID', 'userID'
    """

    ua_cnt = df.groupby(['userID', 'advertiserID']).size().reset_index()
    ua_cnt.rename(columns={0: 'ua_cnt'}, inplace=True)
    ua_cnt = ua_cnt[['userID', 'advertiserID', 'ua_cnt']]
    df = pd.merge(df, ua_cnt, how='left', on=['userID', 'advertiserID'])

    sorted = df.sort_values(by=['userID', 'advertiserID', 'clickTime'], ascending=True)
    first = sorted.drop_duplicates(['userID', 'advertiserID'], keep='first')
    last  = sorted.drop_duplicates(['userID', 'advertiserID'], keep='last')

    first['ua_first'] = 1
    first = first[['ua_first']]
    df = df.join(first)

    last['ua_last'] = 1
    last = last[['ua_last']]
    df = df.join(last)

    df['trick'] = df.apply(trick, axis=1)
    return df

def add_diff(df):
    """
    ** 新学trick： 根据不同的转化方式，计算转化时间
    """

    sorted = df.sort_values(by=['userID', 'advertiserID', 'clickTime'], ascending=True)
    first = sorted.groupby(['userID', 'advertiserID'])['clickTime'].first().reset_index()
    first.rename(columns={'clickTime': 'first_diff'}, inplace=True)
    last = sorted.groupby(['userID', 'advertiserID'])['clickTime'].last().reset_index()
    last.rename(columns={'clickTime': 'last_diff'}, inplace=True)
    df = pd.merge(df, first, 'left', on=['userID', 'advertiserID'])
    df = pd.merge(df, last, 'left', on=['userID', 'advertiserID'])
    df['first_diff'] = df['clickTime'] - df['first_diff']
    df['last_diff'] = df['last_diff'] - df['clickTime']
    return df

def add_install2click(df, i, actions):
    """
    计算CTIT
    """
    install2click = actions[actions.installTime < i * 1000000]
    df = pd.merge(df, install2click, 'left', ['userID', 'appID'])
    df['install2click'] = df['clickTime'] - df['installTime']
    return df

def gen_tricks(start_day, end_day):
    '''
    生成trick,first_diff,last_diff，install2click，根据gloabl_index拼接
    :param start_day:
    :param end_day:
    :return:
    '''
    """
    将新构造的feature进行pickle化
    """
    train_data = load_pickle(raw_data_path + 'train.pkl')
    test_data = load_pickle(raw_data_path + 'test.pkl')
    actions = load_pickle(raw_data_path + 'user_app_actions.pkl')
    data = train_data.append(test_data)
    del train_data, test_data
    data = addTime(data)
    data = addAd(data)

    for day in tqdm(np.arange(start_day, end_day+1)):
        feature_path = feature_data_path + 'tricks_day_' + str(day) + '.pkl'
        if os.path.exists(feature_path):
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            df = data.loc[data.clickDay == day]
            df = add_trick(df)
            df = add_diff(df)
            df = add_install2click(df, day, actions)
            dump_pickle(df[['global_index', 'trick', 'first_diff', 'last_diff', 'install2click']], feature_path)

def add_tricks(data):
    '''
    :param data:
    :return:
    '''
    """
    - 将每一天的feature数据load_pickle出来
    - 和原始的data_frame进行merge后，返回
    - do next step with new data_frame
    """

    tricks = None
    for day in tqdm((data.clickTime // 1000000).unique()):
        feature_path = feature_data_path + 'tricks_day_' + str(day) + '.pkl'
        if not os.path.exists(feature_path):
            gen_tricks(day, day)
        day_tricks = load_pickle(feature_path)
        if tricks is None:
            tricks = day_tricks
        else:
            tricks = pd.concat([tricks, day_tricks], axis=0)
    data = pd.merge(data, tricks, 'left', 'global_index')
    return data

if __name__ == '__main__':
    gen_tricks(23, 31)
    print('All done')

