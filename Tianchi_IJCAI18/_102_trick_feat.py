#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: bettenW
@Github: https://github.com/bettenW
@Desc  : 时间差特征在这次比赛中也算是 trick 的存在：
        - 从用户点击商品的时间差来反映用户购买商品的可能性, 短时间内点击相同商品购买的可能性会比较大. 
        - 从单特征, 多特征进行组合构造
        - 从全局, 天统计首次点击与当前点击的时间差, 最后次点击与当前点击的时间差
        - 上一次点击和下一次点击与当前点击的时间差。
        - 疑问：算出时间差后，后续如何处理这个时间差特征？分桶 or 归一化？
"""

import pandas as pd
import numpy as np
import time
import datetime
import gc
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

def pre_process(data):
    cols = data.columns.tolist()
    keys = ['instance_id', 'day']
    for k in keys:
        cols.remove(k)

    return data, cols
    
def doTrick1(data):
    data.sort_values(['user_id', 'context_timestamp'], inplace=True)
    
    """
    每一个subset都是在处理高阶交互的trick特征
    """

    subset = ['user_id', 'day']
    data['click_user_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_lab'] = 3
    del pos
    gc.collect()

    subset = ['item_id', 'user_id', 'day']
    data['click_user_item_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_item_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_item_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_item_lab'] = 3
    del pos
    gc.collect()

    subset = ['item_brand_id','user_id', 'day']
    data['click_user_brand_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_brand_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_brand_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_brand_lab'] = 3
    del pos
    gc.collect()

    subset = ['shop_id','user_id', 'day']
    data['click_user_shop_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_shop_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_shop_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_shop_lab'] = 3
    del pos
    gc.collect()

    subset = ['item_city_id','user_id', 'day']
    data['click_user_city_lab'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_city_lab'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_city_lab'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'click_user_city_lab'] = 3
    del pos
    gc.collect()


    return data

"""
新学trick: 
    - 单特征和组合特征，加上天级时间窗口，统计当前点击和首次点击和末次点击的时间差，用以反应用户的行为特质。
    - 待优化的点：函数封装的不够彻底，理想状况下是：传入data_frame、subset。本代码中的冗余较多。
"""
def doTrick2(data):
    data.sort_values(['user_id', 'context_timestamp'], inplace=True)

    # user_id
    subset = ['user_id', 'day']
    temp = data.loc[:,['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_first'] = data['context_timestamp'] - data['u_day_diffTime_first'] # 当前点击和天级首次点击的时间差
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset) # 做left_join，保证接下来可以做减法运算
    data['u_day_diffTime_last'] = data['u_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    # 如果当天没有重复点击，即不存在多次点击时，置为-1
    data.loc[~data.duplicated(subset=subset, keep=False), ['u_day_diffTime_first', 'u_day_diffTime_last']] = -1

    # item_id
    subset = ['item_id', 'day']
    temp = data.loc[:,['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_first'] = data['context_timestamp'] - data['i_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_last'] = data['i_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['i_day_diffTime_first', 'i_day_diffTime_last']] = -1

    # item_brand_id, user_id
    subset = ['item_brand_id', 'user_id', 'day']
    temp = data.loc[:,['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'b_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['b_day_diffTime_first'] = data['context_timestamp'] - data['b_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'b_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['b_day_diffTime_last'] = data['b_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['b_day_diffTime_first', 'b_day_diffTime_last']] = -1
    
    # shop_id, user_id
    subset = ['shop_id', 'user_id', 'day']
    temp = data.loc[:,['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 's_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['s_day_diffTime_first'] = data['context_timestamp'] - data['s_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:,['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 's_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['s_day_diffTime_last'] = data['s_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['s_day_diffTime_first', 's_day_diffTime_last']] = -1

    return data
"""
新学trick：
    - 单特征的时间差特征构建：包括距离上一次点击的时间差和距离下一次点击的时间差
    - 本脚本仅基于'user_id', 'item_id'
    - 可以考虑使用组合特征，添加item相关和user相关的其它特征进行交叉。
"""
def lasttimeDiff(data):
    for column in ['user_id', 'item_id']:
        """
        这里最好增加一个排序，按照context_timestamp升序排列
        """
        gc.collect()
        data[column+'_lasttime_diff'] = 0
        train_data = data[['context_timestamp', column, column+'_lasttime_diff']].values
        lasttime_dict = {}
        """
        复习一下：
        这里是通过loop，不断地对df_list[2]的值进行覆盖。
        """
        for df_list in train_data:
            if df_list[1] not in lasttime_dict:
                df_list[2] = -1
                lasttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
                lasttime_dict[df_list[1]] = df_list[0]
        data[['context_timestamp', column, column+'_lasttime_diff']] = train_data
    return data

def nexttimeDiff(data):
    for column in ['user_id', 'item_id']:
        gc.collect()
        data[column+'_nexttime_diff'] = 0
        train_data = data[['context_timestamp', column, column+'_nexttime_diff']].values
        nexttime_dict = {}
        for df_list in train_data:
            if df_list[1] not in nexttime_dict:
                df_list[2] = -1
                nexttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
                nexttime_dict[df_list[1]] = df_list[0]
        data[['context_timestamp', column, column+'_nexttime_diff']] = train_data

    return data

def main():
    path = './data/'
    
    train = pd.read_csv(path+'train_all.csv')
    test = pd.read_csv(path+'test_all.csv')

    data = train.append(test, ignore_index=True)

    data, cols = pre_process(data)
    print('pre_process data:', data.shape)

    ###########挖掘新的特征###########

    # 对不同点击进行标记
    data = doTrick1(data)
    print('doTrick1 data:', data.shape)

    # 同一天点击时间差
    data = doTrick2(data)
    print('doTrick2 data:', data.shape)
    
    # 单特征距离上一次点击时间差
    data = lasttimeDiff(data)
    print('lasttimeDiff data:', data.shape)
    
    # 单特征距离下一次点击时间差
    data = nexttimeDiff(data)
    print('lasttimeDiff data:', data.shape)

    ############挖掘新的特征###########

    data = data.drop(cols, axis=1)

    # 得到全部训练集
    print('经过处理后,全部训练集最终维度:', data.shape)
    data.to_csv(path+'102_trick_feat_all.csv', index=False)

    # 得到7号训练集
    data = data.loc[data.day == 7]
    data = data.drop('day', axis=1)
    print('经过处理后,7号数据集最终维度:', data.shape)
    print(data.columns.tolist())
    data.to_csv(path+'102_trick_feat.csv', index=False)
    

if __name__ == '__main__':
    main()
