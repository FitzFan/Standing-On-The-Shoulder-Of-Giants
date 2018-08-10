#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
"""

import os
import zipfile
import time
import pickle
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
import  xgboost as xgb
import lightgbm as lgb

from _2_7_gen_trick_final import add_click_trick
from _3_0_gen_final_data import gen_offline_data,gen_online_data
from utils import load_pickle,dump_pickle,get_feature_value,feature_spearmanr,feature_target_spearmanr,addCrossFeature,calibration
from utils import raw_data_path,feature_data_path,cache_pkl_path,result_path,analyse

get_ipython().magic(u'matplotlib inline')


# # XGB 4天

# In[3]:


train_x,test_x = gen_online_data(25,29,31)


# In[4]:


feature_group_A = ['creativeID', 'userID',
       'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
       'education', 'marriageStatus', 'haveBaby', 'ht_province',
       'rd_province', 'sitesetID', 'positionType', 'adID',
       'camgaignID', 'advertiserID', 'appID', 'appPlatform',
     # 'user_start_install_cate_0',
                 'user_start_install_cate_1',
       'user_start_install_cate_2', 'user_start_install_cate_3',
       'user_start_install_cate_4', 'user_start_install_cate_5',
                 
       'appCategory', 'trick', 'first_diff', 'last_diff', 'user_hist_install', 'clickHour',
        'user_adID_click_day', 'user_adID_click_hour',
       'user_camgaignID_click_day', 'user_camgaignID_click_hour',
       'user_appID_click_day', 'user_appID_click_hour',

                 
                 'user_sitesetID_click_day',
       'user_sitesetID_click_hour', 'user_click_day',
                 
        'positionID_cvr_smooth','creativeID_cvr','userID_cvr','adID_cvr','appID_cvr',
     'positionID_sum_count', 'creativeID_sum_count', 'appID_sum_count',
       'adID_sum_count', 'userID_sum_count',
        'user_adID_click_day_mean', 'user_adID_click_day_min',
       'user_camgaignID_click_day_min', 'user_appID_click_day_mean',
       'user_appID_click_day_max', 'user_appID_click_day_min',
       'user_sitesetID_click_day_mean', 'user_sitesetID_click_day_max',
       'user_sitesetID_click_day_min', 'user_click_day_mean', 'user_click_day_max','user_click_day_min',
        'advertiserID_user_clicks_age_cut_0',
       'advertiserID_user_clicks_age_cut_1',
       'advertiserID_user_clicks_age_cut_2',
       'advertiserID_user_clicks_age_cut_3',
       'advertiserID_user_clicks_age_cut_4',
       'advertiserID_user_clicks_age_cut_5',
       'advertiserID_user_clicks_age_cut_6',
     # 'advertiserID_user_clicks_age_cut_7',
       'advertiserID_user_clicks_gender_0',
       'advertiserID_user_clicks_gender_1',
       'advertiserID_user_clicks_gender_2',
       'advertiserID_user_clicks_education_0',
       'advertiserID_user_clicks_education_1',
       'advertiserID_user_clicks_education_2',
       'advertiserID_user_clicks_education_3',
       'advertiserID_user_clicks_education_4',
       'advertiserID_user_clicks_education_5',
       'advertiserID_user_clicks_education_6',
      # 'advertiserID_user_clicks_education_7',
       'advertiserID_user_clicks_marriageStatus_0',
       'advertiserID_user_clicks_marriageStatus_1',
       'advertiserID_user_clicks_marriageStatus_2',
       'advertiserID_user_clicks_marriageStatus_3',
        
       'appID_user_clicks_age_cut_0', 'appID_user_clicks_age_cut_1',
       'appID_user_clicks_age_cut_2', 'appID_user_clicks_age_cut_3',
       'appID_user_clicks_age_cut_4', 'appID_user_clicks_age_cut_5',
       'appID_user_clicks_age_cut_6',
                   #'appID_user_clicks_age_cut_7',
       'appID_user_clicks_gender_0', 'appID_user_clicks_gender_1',
       'appID_user_clicks_gender_2', 'appID_user_clicks_education_0',
       'appID_user_clicks_education_1', 'appID_user_clicks_education_2',
       'appID_user_clicks_education_3', 'appID_user_clicks_education_4',
       'appID_user_clicks_education_5', 'appID_user_clicks_education_6',
       #'appID_user_clicks_education_7',
                   'appID_user_clicks_marriageStatus_0',
       'appID_user_clicks_marriageStatus_1',
       'appID_user_clicks_marriageStatus_2',
       'appID_user_clicks_marriageStatus_3', 
                 'appID_user_clicks_haveBaby_0',
       'appID_user_clicks_haveBaby_1', 'appID_user_clicks_haveBaby_2',
       'appID_user_clicks_haveBaby_3', 'appID_user_clicks_haveBaby_4',
       'appID_user_clicks_haveBaby_5', 'appID_user_clicks_haveBaby_6',]
print(len(feature_group_A))


# In[8]:


dtrain = xgb.DMatrix(train_x.loc[train_x.clickDay>=26,feature_group_A].values, train_x.loc[train_x.clickDay>=26,'label'].values, feature_names=feature_group_A)
del train_x
dtest=xgb.DMatrix(test_x[feature_group_A].values,feature_names=feature_group_A)
del test_x
watchlist = [(dtrain, 'train'), (dtrain, 'val')]
params={
    'max_depth':8,
    'nthread':25,
    'eta':0.1,
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'subsample':0.7,
    'colsample_bytree':0.5,
    'silent':1,
    'seed':1123,
    'min_child_weight':10
    #'scale_pos_weight':0.5
}

xgb_a=xgb.train(params,dtrain,
              num_boost_round=140,
              early_stopping_rounds=20,
              evals=watchlist,
              verbose_eval=10)
del dtrain
xgb_a_ans = xgb_a.predict(dtest)
del dtest


# # XGB 5天

# In[9]:


feature_group_B= [#'creativeID', 'userID',
                 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
                 'education', 'marriageStatus', 'haveBaby', 'ht_province',
                 'rd_province', 'sitesetID', 'positionType', 'adID',
                 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
                 # 'user_start_install_cate_0',
                 'user_start_install_cate_1',
                 'user_start_install_cate_2', 'user_start_install_cate_3',
                 'user_start_install_cate_4', 'user_start_install_cate_5',

                 'appCategory', 'trick', 'first_diff', 'last_diff', 'user_hist_install', 'clickHour',
                 'user_adID_click_day', 'user_adID_click_hour',
                 'user_camgaignID_click_day', 'user_camgaignID_click_hour',
                 'user_appID_click_day', 'user_appID_click_hour',

                 'user_sitesetID_click_day',
                 'user_sitesetID_click_hour', 'user_click_day',

                 'positionID_cvr_smooth', 'creativeID_cvr', 'userID_cvr', 'adID_cvr', 'appID_cvr',
                 'positionID_sum_count', 'creativeID_sum_count', 'appID_sum_count',
                 'adID_sum_count', 'userID_sum_count',
                 'user_adID_click_day_mean', 'user_adID_click_day_min',
                 'user_camgaignID_click_day_min', 'user_appID_click_day_mean',
                 'user_appID_click_day_max', 'user_appID_click_day_min',
                 'user_sitesetID_click_day_mean', 'user_sitesetID_click_day_max',
                 'user_sitesetID_click_day_min', 'user_click_day_mean', 'user_click_day_max', 'user_click_day_min',
                 'advertiserID_user_clicks_age_cut_0',
                 'advertiserID_user_clicks_age_cut_1',
                 'advertiserID_user_clicks_age_cut_2',
                 'advertiserID_user_clicks_age_cut_3',
                 'advertiserID_user_clicks_age_cut_4',
                 'advertiserID_user_clicks_age_cut_5',
                 'advertiserID_user_clicks_age_cut_6',
                 #'advertiserID_user_clicks_age_cut_7',
                 'advertiserID_user_clicks_gender_0',
                 'advertiserID_user_clicks_gender_1',
                 'advertiserID_user_clicks_gender_2',
                 'advertiserID_user_clicks_education_0',
                 'advertiserID_user_clicks_education_1',
                 'advertiserID_user_clicks_education_2',
                 'advertiserID_user_clicks_education_3',
                 'advertiserID_user_clicks_education_4',
                 'advertiserID_user_clicks_education_5',
                 'advertiserID_user_clicks_education_6',
                # 'advertiserID_user_clicks_education_7',
                 'advertiserID_user_clicks_marriageStatus_0',
                 'advertiserID_user_clicks_marriageStatus_1',
                 'advertiserID_user_clicks_marriageStatus_2',
                 'advertiserID_user_clicks_marriageStatus_3',

                 'appID_user_clicks_age_cut_0', 'appID_user_clicks_age_cut_1',
                 'appID_user_clicks_age_cut_2', 'appID_user_clicks_age_cut_3',
                 'appID_user_clicks_age_cut_4', 'appID_user_clicks_age_cut_5',
                 'appID_user_clicks_age_cut_6',
    #'appID_user_clicks_age_cut_7',
                 'appID_user_clicks_gender_0', 'appID_user_clicks_gender_1',
                 'appID_user_clicks_gender_2', 'appID_user_clicks_education_0',
                 'appID_user_clicks_education_1', 'appID_user_clicks_education_2',
                 'appID_user_clicks_education_3', 'appID_user_clicks_education_4',
                 'appID_user_clicks_education_5', 'appID_user_clicks_education_6',
                # 'appID_user_clicks_education_7',
    'appID_user_clicks_marriageStatus_0',
                 'appID_user_clicks_marriageStatus_1',
                 'appID_user_clicks_marriageStatus_2',
                 'appID_user_clicks_marriageStatus_3',
                 'appID_user_clicks_haveBaby_0',
                 'appID_user_clicks_haveBaby_1', 'appID_user_clicks_haveBaby_2',
                 'appID_user_clicks_haveBaby_3', 'appID_user_clicks_haveBaby_4',
                 'appID_user_clicks_haveBaby_5', 'appID_user_clicks_haveBaby_6','install2click']


# In[10]:


X_train = train_x[feature_group_A].values
X_test = test_x[feature_group_B].values
y = train_x['label'].values
del train_x,test_x

params={
    'max_depth':10,
    'nthread':25,
    'eta':0.1,
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'subsample':0.8,
    'colsample_bytree':0.7,
    'silent':1,
    'seed':1123,
    'min_child_weight':10
    #'scale_pos_weight':0.5
}

dtrain = xgb.DMatrix(X_train, y)
watchlist = [(dtrain, 'train'), ]

xgb_b = xgb.train(params, dtrain,
                num_boost_round=350,
                early_stopping_rounds=20,
                evals=watchlist,
                verbose_eval=10)

dtest = xgb.DMatrix(X_test)
xgb_b_ans = xgb_b.predict(dtest)


# In[ ]:


xgb_result = xgb_b_ans * 0.6 + xgb_a_ans * 0.4
result = pd.read_csv('../result/demo_result.csv',index_col=['instanceID'])
result['prob'] = xgb_result
result.to_csv(result_path+'submission_xgb.csv')

