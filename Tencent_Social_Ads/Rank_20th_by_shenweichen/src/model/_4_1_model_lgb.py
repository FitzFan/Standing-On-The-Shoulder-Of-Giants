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
import lightgbm as lgb

from _2_7_gen_trick_final import add_click_trick
from _3_0_gen_final_data import gen_offline_data,gen_online_data
from utils import load_pickle,dump_pickle,get_feature_value,feature_spearmanr,feature_target_spearmanr,addCrossFeature,calibration
from utils import raw_data_path,feature_data_path,cache_pkl_path,result_path,analyse

get_ipython().magic(u'matplotlib inline')


# In[ ]:


def training(clf,train_x,test_x,feature_names,cate_features,mode='offline'):
    if mode=='offline':
        start_time = time.time()
        clf.fit(train_x[feature_names],train_x['label'],
               eval_set=[(train_x[feature_names],train_x['label']),(test_x[feature_names],test_x['label'])],
                 feature_name=feature_names,categorical_feature=cate_features,
                    early_stopping_rounds=20,
                    verbose=10,
           )
        total_time = time.time()-start_time
        print('offline training done {0}m{1:.1f}s'.format(total_time//60,total_time%60))
        print('best iteration {0}'.format(clf.best_iteration))
        print('best score {0:.6f}'.format(clf.best_score['valid_1']['binary_logloss']))
        #pred = clf.predict_proba(test_x.loc[:,feature_names],num_iteration=clf.best_iteration)[:,1]
        #print('%.7f'%log_loss(test_x.loc[:,'label'],pred))
    
    elif mode=='online':
        start_time = time.time()
        clf.fit(train_x[feature_names],train_x['label'],
               eval_set=[(train_x[feature_names],train_x['label'])],
                 feature_name=feature_names,categorical_feature=cate_features,
                    #early_stopping_rounds=20,
                    verbose=10,
           )
        total_time = time.time()-start_time
        print('online training done {0}m{1:.1f}s'.format(total_time//60,total_time%60))
    return clf

def gen_result(clf,test_x,feature_names):
    print('start predicting...')
    start_time = time.time()
    test_prob = clf.predict_proba(test_x[feature_names])[:,1]
    print(test_prob.mean())
    result = pd.read_csv('../result/demo_result.csv',index_col=['instanceID'])
    result['prob'] = test_prob
    filename = 'submission_'+'_'.join(time.ctime()[4:16].replace(':',' ').split(' '))+'.zip'
    cali_name = 'submission_'+'_'.join(time.ctime()[4:16].replace(':',' ').split(' '))+'.cali.zip'
    result.to_csv('submission.csv')
    print(filename)
    with zipfile.ZipFile('../result/'+filename, "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)
    total_time = time.time() - start_time
    print('online predicting time {0}m{1:.1f}s'.format(total_time//60,total_time%60))
    return cali_name
    


# # LightGBM model  
# - lgb 4天 feature_group_A -- ans_a
# - lgb 4天 feature_group_B -- ans_b
# - lgb 5天 feature_group_B -- ans_c  
# $ LGB_{result} =  ans_a*0.3 + ans_b*0.1+ans_c*0.6$

# In[ ]:


feature_group_A = [
                 'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
                 'education', 'marriageStatus', 'haveBaby', 'ht_province',
                 'rd_province', 'sitesetID', 'positionType', 'adID',
                 'camgaignID', 'advertiserID', 'appID', 'appPlatform',
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
                 'advertiserID_user_clicks_marriageStatus_0',
                 'advertiserID_user_clicks_marriageStatus_1',
                 'advertiserID_user_clicks_marriageStatus_2',
                 'advertiserID_user_clicks_marriageStatus_3',
                 'appID_user_clicks_age_cut_0', 'appID_user_clicks_age_cut_1',
                 'appID_user_clicks_age_cut_2', 'appID_user_clicks_age_cut_3',
                 'appID_user_clicks_age_cut_4', 'appID_user_clicks_age_cut_5',
                 'appID_user_clicks_age_cut_6',
                 'appID_user_clicks_gender_0', 'appID_user_clicks_gender_1',
                 'appID_user_clicks_gender_2', 'appID_user_clicks_education_0',
                 'appID_user_clicks_education_1', 'appID_user_clicks_education_2',
                 'appID_user_clicks_education_3', 'appID_user_clicks_education_4',
                 'appID_user_clicks_education_5', 'appID_user_clicks_education_6',
    			 'appID_user_clicks_marriageStatus_0',
                 'appID_user_clicks_marriageStatus_1',
                 'appID_user_clicks_marriageStatus_2',
                 'appID_user_clicks_marriageStatus_3',
                 'appID_user_clicks_haveBaby_0','install2click','global_uct_cnt','global_first',
                 'appID_user_clicks_haveBaby_1', 'appID_user_clicks_haveBaby_2',
                 'appID_user_clicks_haveBaby_3', 'appID_user_clicks_haveBaby_4',
                 'appID_user_clicks_haveBaby_5', 'appID_user_clicks_haveBaby_6','global_last']

feature_group_B = ['creativeID', 'userID',
					'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
					'education', 'marriageStatus', 'haveBaby', 'ht_province',
					'rd_province', 'sitesetID', 'positionType', 'adID',
					'camgaignID', 'advertiserID', 'appID', 'appPlatform',
					'user_start_install_cate_0','user_start_install_cate_1',
					'user_start_install_cate_2', 'user_start_install_cate_3',
					'user_start_install_cate_4', 'user_start_install_cate_5',         
					'appCategory', 'trick', 'first_diff', 'last_diff', 'user_hist_install', 'clickHour',
					'user_adID_click_day', 'user_adID_click_hour',
					'user_camgaignID_click_day', 'user_camgaignID_click_hour',
					'user_appID_click_day', 'user_appID_click_hour','user_sitesetID_click_day',
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
					'advertiserID_user_clicks_marriageStatus_0',
					'advertiserID_user_clicks_marriageStatus_1',
					'advertiserID_user_clicks_marriageStatus_2',
					'advertiserID_user_clicks_marriageStatus_3',
					'advertiserID_user_clicks_haveBaby_0',
					'advertiserID_user_clicks_haveBaby_1',
					'advertiserID_user_clicks_haveBaby_2',
					'advertiserID_user_clicks_haveBaby_3',
					'advertiserID_user_clicks_haveBaby_4',
					'advertiserID_user_clicks_haveBaby_5',
					'advertiserID_user_clicks_haveBaby_6', 
					'appID_user_clicks_age_cut_0', 'appID_user_clicks_age_cut_1',
					'appID_user_clicks_age_cut_2', 'appID_user_clicks_age_cut_3',
					'appID_user_clicks_age_cut_4', 'appID_user_clicks_age_cut_5',
					'appID_user_clicks_age_cut_6',# 'appID_user_clicks_age_cut_7',
					'appID_user_clicks_gender_0', 'appID_user_clicks_gender_1',
					'appID_user_clicks_gender_2', 'appID_user_clicks_education_0',
					'appID_user_clicks_education_1', 'appID_user_clicks_education_2',
					'appID_user_clicks_education_3', 'appID_user_clicks_education_4',
					'appID_user_clicks_education_5', 'appID_user_clicks_education_6',
					'appID_user_clicks_marriageStatus_0',
					'appID_user_clicks_marriageStatus_1',
					'appID_user_clicks_marriageStatus_2',
					'appID_user_clicks_marriageStatus_3', 
					'appID_user_clicks_haveBaby_0',
					'appID_user_clicks_haveBaby_1', 'appID_user_clicks_haveBaby_2',
					'appID_user_clicks_haveBaby_3', 'appID_user_clicks_haveBaby_4',
					'appID_user_clicks_haveBaby_5', 'appID_user_clicks_haveBaby_6',
					'install2click']
cate_features = []
print('featureA numbers',len(feature_group_A))
print('featureB numbers',len(feature_group_B))


train_x,test_x = gen_online_data(25,29,31)
train_x = add_click_trick(train_x,25,29)
test_x = add_click_trick(test_x,31,31)

lgb_a = LGBMClassifier(num_leaves=110, max_depth=12,
                     learning_rate=0.1, n_estimators=1200,
                     seed=0, nthread=24, subsample=0.8, colsample_bytree=0.9,
                     reg_lambda=0.005, )
lgb_a.fit(train_x.loc[train_x.clickDay>25,feature_group_A],train_x.loc[train_x.clickDay>25,['label']].values,
        eval_set=[(train_x.loc[train_x.clickDay>25,feature_group_A],train_x.loc[train_x.clickDay>25,['label']])],
             feature_name=feature_group_A,categorical_feature=cate_features,
            verbose=50,
           )
lgb_a_ans = lgb_a.predict_proba(test_x[feature_group_A],num_iteration=1200)[:,1]#0.102706



lgb_b = LGBMClassifier(num_leaves=110, max_depth=12,
                     learning_rate=0.03, n_estimators=1350,
                     seed=0, nthread=20, subsample=0.8, colsample_bytree=0.9,
                     reg_lambda=0.005, )
lgb_b.fit(train_x.loc[train_x.clickDay>25,feature_group_B],train_x.loc[train_x.clickDay>25,['label']].values,
             feature_name=feature_group_B,categorical_feature=cate_features,
            verbose=50,
           )
lgb_b_ans = lgb_b.predict_proba(test_x[feature_group_B],num_iteration=1305)[:,1] #0.102339

lgb_c = LGBMClassifier(num_leaves=110, max_depth=12,
                     learning_rate=0.03, n_estimators=1500,
                     seed=0, nthread=24, subsample=0.8, colsample_bytree=0.9,
                     reg_lambda=0.005, )
lgb_c.fit(train_x.loc[train_x.clickDay>=25,feature_group_B],train_x.loc[train_x.clickDay>=25,['label']].values,
        eval_set=[(train_x.loc[train_x.clickDay>=25,feature_group_B],train_x.loc[train_x.clickDay>=25,['label']])],
             feature_name=feature_group_B,categorical_feature=cate_features,
            verbose=50,
           )
lgb_c_ans = lgb_c.predict_proba(test_x[feature_group_B],num_iteration=1500)[:,1]

lgb_result = lgb_a_ans*0.3 + lgb_b_ans*0.1+lgb_c_ans*0.6
result = pd.read_csv('../result/demo_result.csv',index_col=['instanceID'])
result['prob'] = lgb_result
result.to_csv(result_path+'submission_lgb.csv')

