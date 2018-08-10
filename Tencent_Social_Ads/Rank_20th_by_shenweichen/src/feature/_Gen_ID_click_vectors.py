#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Gen_ID_click_vectors.py
@time: 2018/3/19 17:16
@desc: 广告描述向量特征提取
       广告投放是有特定受众对象的，而特定的受众对象也可以描述广告的相关特性，
       使用不同的人口属性对广告ID和APPID进行向量表示，学习隐含的语义特征。
       **这个脚本是针对的广告投放的数据，与“Gen_app_install_features.py”的分析对象不一样：
        - 本脚本的分析对象是train.pkl and test.pkl
        - Gen_app_install_features.py的分析对象installed.pkl and others
        - 二者代码的思路是一样的：
            - 先使用user_id和受众属性特征进行one-hot处理，产出temp_frame
            - 对raw_frame和temp_frame，基于user_id进行left_join
            - 基于ID_name + dummy_features基于聚合，算子为sum()
            - 移除多余的dummy_feature列
            - pickle_dump()
"""

import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from Ad_Utils import load_pickle, dump_pickle, raw_data_path, feature_data_path
from Feature_joint import addTime, addPosition, addAd, addAppCategories, addUserInfo
from sklearn.feature_extraction.text import TfidfTransformer


def gen_CountVector_ID_user_clicks(ID_name, last_day=27, ID_describe_feature_names=None, drop_na=False):
    '''
    生成根据train和test表计算的ID_name计数描述向量
    拼接键 [ID_name]
    :param ID_name:
    :param last_day:
    :param gen_CountVector_ID_user_clicks:
    :param drop_na:
    :return:
    '''
    if ID_describe_feature_names is None:
        ID_describe_feature_names = ['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby']

    train = load_pickle(raw_data_path + 'train.pkl')
    test = load_pickle(raw_data_path + 'test.pkl')
    data = train.append(test)
    data = addTime(data)
    data = data[data.clickDay <= last_day]
    data = addAd(data)
    data = addPosition(data)
    data = addAppCategories(data)
    data = data[['userID', ID_name]]
    user_info = pd.read_csv(raw_data_path + 'user.csv')

    user_info['age_cut'] = pd.cut(user_info['age'], bins=[-1, 0, 18, 25, 35, 45, 55, np.inf], labels=False)
    user_info.loc[user_info.education == 7, 'education'] = 6

    user_info['hometown_province'] = user_info['hometown'].apply(lambda x: x // 100)
    user_info['residence_province'] = user_info['residence'].apply(lambda x: x // 100)

    for feature in tqdm(ID_describe_feature_names):
        feature_path = feature_data_path + 'CountVector_' + ID_name + '_user_clicks_' + feature + '_lastday' + str(last_day) + '.pkl'
        if drop_na:
            feature_path += '.no_na'
        if os.path.exists(feature_path):
            print('found ' + feature_path)
            continue
        print('generating ' + feature_path)
        prefix_name = ID_name + '_user_clicks_' + feature
        sub_user_info = pd.get_dummies(user_info[['userID', feature]], columns=[feature], prefix=prefix_name)
        if drop_na:
            sub_user_info.drop([prefix_name + '_0'], axis=1, inplace=True)
        data = pd.merge(data, sub_user_info, 'left', 'userID') # 虽然按照参数顺序传参是可以，但感觉还是用how='left', on='userID'的形式传参更为清晰
        dummy_features = sub_user_info.columns.tolist()
        dummy_features.remove('userID')
        ID_describe_feature = data[[ID_name] + dummy_features].groupby([ID_name], as_index=False).sum()
        data.drop(dummy_features, axis=1, inplace=True)
        dump_pickle(ID_describe_feature, feature_path)

def get_ConcatedTfidfVector_ID_user_clicks(ID_name, last_day, mode='local', concated_list=None, drop_na=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
    """
    使用默认的local模式效果稍微好一些
    测试过advertiserID, camgaignID, adID, creativeID, appID, appCategory, cate_A, appPlatform, positionType
    advertiserID效果较好，appID效果其次,然后是appCategory，其他都不好
    所以main()中的参数传递只有那几个
    """
    """
    - 将每一天的feature数据load_pickle出来
    - 对中间数据进行tf-idf
    - 返回tf-idf的frame

    PS:为什么不直接对tf-idf的数据进行pickle？
        - 因为不确定是否tf-idf对最后的模型训练带来正向结果。
    """

    if concated_list is None:
        concated_list = ['age_cut', 'gender', 'education', 'marriageStatus','haveBaby']

    tfidf_vec = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    concated_tfidf_vec = None

    for feature in tqdm(concated_list):
        feature_path = feature_data_path + 'CountVector_' + ID_name + '_user_clicks_' + feature + '_lastday' + str(last_day) + '.pkl'
        if drop_na:
            feature_path += '.no_na'
        if not os.path.exists(feature_path):
            gen_CountVector_ID_user_clicks(ID_name)
        count_vec = load_pickle(feature_path)
        if mode == 'local':
            count_vec.set_index(ID_name, inplace=True)
            vec_columns = count_vec.columns
            local_tfidf_vec = tfidf_vec.fit_transform(count_vec).todense()
            local_tfidf_vec = pd.DataFrame(local_tfidf_vec, columns=vec_columns, index=count_vec.index).reset_index()
        elif mode == 'global':
            local_tfidf_vec = count_vec

        if concated_tfidf_vec is None:
            concated_tfidf_vec = local_tfidf_vec
        else:
            concated_tfidf_vec = pd.merge(concated_tfidf_vec, local_tfidf_vec, 'left', ID_name)
    if mode == 'global':
        concated_tfidf_vec.set_index(ID_name, inplace=True)
        vec_columns = concated_tfidf_vec.columns
        global_concated_tfidf_vec = tfidf_vec.fit_transform(concated_tfidf_vec).todense()
        global_concated_tfidf_vec = pd.DataFrame(global_concated_tfidf_vec, columns=vec_columns, index=concated_tfidf_vec.index)
        concated_tfidf_vec = global_concated_tfidf_vec.reset_index()
    return concated_tfidf_vec

def main():
    gen_CountVector_ID_user_clicks('advertiserID', 31)
    gen_CountVector_ID_user_clicks('appID', 31)
    gen_CountVector_ID_user_clicks('advertiserID', 27)
    gen_CountVector_ID_user_clicks('appID', 27)
    print('All done')

if __name__ == '__main__':
    main()