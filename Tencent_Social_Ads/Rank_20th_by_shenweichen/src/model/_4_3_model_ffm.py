#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
"""

import pandas as pd
import numpy as np
import zipfile
import subprocess

from common import data2libffm
from _3_0_gen_final_data import gen_offline_data,gen_online_data
from utils import calibration,cache_pkl_path,result_path


# In[4]:


train,test = gen_online_data(25,29,31)


# In[5]:


def binning(series, bin_num):
    bins = np.linspace(series.min(), series.max(), bin_num)
    labels = [i for i in range(bin_num-1)]
    out = pd.cut(series, bins=bins, labels=labels).astype(float)
    return out

# 纯cate
categorical_field = ['creativeID', 'userID',
       'positionID', 'connectionType', 'telecomsOperator', 'age', 'gender',
       'education', 'marriageStatus', 'haveBaby', 'ht_province',
       'rd_province', 'sitesetID', 'positionType', 'adID',
       'camgaignID', 'advertiserID', 'appID', 'appPlatform',
       'appCategory', 'trick', 'clickHour',]

#连续log平方
continue_field1 = ['first_diff', 'last_diff','install2click',
                    'positionID_sum_count', 'creativeID_sum_count',
       'appID_sum_count', 'adID_sum_count', 'userID_sum_count',]

#连续分箱
continue_field2 = ['positionID_cvr_smooth','creativeID_cvr','userID_cvr','adID_cvr','appID_cvr',
                  'user_hist_install',]

#连续直接当cate
continue_field3 = ['user_start_install_cate_0',
       'user_start_install_cate_1', 'user_start_install_cate_2',
       'user_start_install_cate_3', 'user_start_install_cate_4',
       'user_start_install_cate_5',
                   'user_adID_click_day', 'user_camgaignID_click_day',
                   'user_camgaignID_click_hour','user_appID_click_day', 'user_appID_click_hour', 
                   'user_sitesetID_click_day','user_sitesetID_click_hour', 'user_click_day',
                   'user_adID_click_day_min','user_camgaignID_click_day_min','user_appID_click_day_max',
                   'user_appID_click_day_min','user_sitesetID_click_day_max','user_sitesetID_click_day_min',
                   'user_click_day_max','user_click_day_min',]


#连续取整当cate
continue_field4 = ['user_adID_click_day_mean','user_appID_click_day_mean','user_sitesetID_click_day_mean','user_click_day_mean',]

field = categorical_field + continue_field1 + continue_field2 + continue_field3 + continue_field4
columns = ['label'] + field + ['clickTime']

#先把训练集和测试集拼在一起
train_data = train_data[train_data.clickTime >= 26000000]
train_data = train_data[columns]
test_data = test_data[columns]
test_data['label'] = 0
tt = pd.concat([train_data, test_data], axis=0)
del train_data
del test_data
gc.collect()

for col in continue_field1:
    tt[col] = np.floor(np.log1p(tt[col]) ** 2)
for col in continue_field2:
    tt[col] = binning(tt[col], 51)
for col in continue_field4:
    tt[col] = np.floor(tt[col])
tt['age'] = np.ceil(tt['age'] / 10)

train = tt[(tt.clickTime >= 26000000) & (tt.clickTime < 30000000)]
test = tt[tt.clickTime >= 31000000]
del train['clickTime']
del test['clickTime']
del tt
gc.collect()

data2libffm(train, cache_pkl_path+'online_train.ffm')
data2libffm(test, cache_pkl_path+'online_test.ffm')


# In[ ]:


train_path = cache_pkl_path+'online_train.ffm'
test_path = cache_pkl_path+'online_test.ffm'
model_path = 'ffm.model'
result_path = 'online_pred.csv'

#./ffm-train -r 0.05 -t 23 -s 20 -l 0.0000005 train_path model_path
#./ffm-predict test_path model_path online_pred.csv
subprocess.call('ffm-train -r 0.05 -t 23 -s 20 -l 0.0000005 {0} {1}'.format(train_path,model_path))
subprocess.call('ffm-predict{0} {1} {2}'.format(test_path,model_path,result_path))

ans = pd.read_csv('online_pred.csv',names=['prob'])
result = pd.read_csv('../result/demo_result.csv')
result['prob'] = ans.prob.values
result.to_csv(result_path+'submission_ffm.csv')

