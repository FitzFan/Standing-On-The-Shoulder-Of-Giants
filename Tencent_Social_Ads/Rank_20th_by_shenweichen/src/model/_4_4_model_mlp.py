#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
"""


import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import h5py
import os
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import result_path
from _3_0_gen_final_data import gen_offline_data,gen_online_data


# In[2]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[ ]:


train,test= gen_online_data(25,29,31)


# In[4]:


feature_names = [#'creativeID', 'userID',
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
        'user_ad_click_day_mean', 'user_ad_click_day_min',
       'user_camgaign_click_day_min', 'user_app_click_day_mean',
       'user_app_click_day_max', 'user_app_click_day_min',
       'user_site_click_day_mean', 'user_site_click_day_max',
       'user_site_click_day_min', 'user_click_day_mean', 'user_click_day_max','user_click_day_min',
        'advertiserID_user_clicks_age_cut_0',
       'advertiserID_user_clicks_age_cut_1',
       'advertiserID_user_clicks_age_cut_2',
       'advertiserID_user_clicks_age_cut_3',
       'advertiserID_user_clicks_age_cut_4',
       'advertiserID_user_clicks_age_cut_5',
       'advertiserID_user_clicks_age_cut_6',
      'advertiserID_user_clicks_age_cut_7',
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
       'advertiserID_user_clicks_education_7',
       'advertiserID_user_clicks_marriageStatus_0',
       'advertiserID_user_clicks_marriageStatus_1',
       'advertiserID_user_clicks_marriageStatus_2',
       'advertiserID_user_clicks_marriageStatus_3',
        
       'appID_user_clicks_age_cut_0', 'appID_user_clicks_age_cut_1',
       'appID_user_clicks_age_cut_2', 'appID_user_clicks_age_cut_3',
       'appID_user_clicks_age_cut_4', 'appID_user_clicks_age_cut_5',
       'appID_user_clicks_age_cut_6', 'appID_user_clicks_age_cut_7',
       'appID_user_clicks_gender_0', 'appID_user_clicks_gender_1',
       'appID_user_clicks_gender_2', 'appID_user_clicks_education_0',
       'appID_user_clicks_education_1', 'appID_user_clicks_education_2',
       'appID_user_clicks_education_3', 'appID_user_clicks_education_4',
       'appID_user_clicks_education_5', 'appID_user_clicks_education_6',
       'appID_user_clicks_education_7', 'appID_user_clicks_marriageStatus_0',
       'appID_user_clicks_marriageStatus_1',
       'appID_user_clicks_marriageStatus_2',
       'appID_user_clicks_marriageStatus_3', 
                 'appID_user_clicks_haveBaby_0',
       'appID_user_clicks_haveBaby_1', 'appID_user_clicks_haveBaby_2',
       'appID_user_clicks_haveBaby_3', 'appID_user_clicks_haveBaby_4',
       'appID_user_clicks_haveBaby_5', 'appID_user_clicks_haveBaby_6','install2click']


# In[19]:


import gc
gc.collect()


# In[ ]:


X_train = train[feature_names].fillna(0).values
X_test = test[feature_names].fillna(0).values
y = train['label'].values
y_test = test['label'].values
del train,test


# In[22]:


print(X_train.shape,X_test.shape)


# In[23]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
st = MinMaxScaler()
st.fit(X_train)
X_train = st.transform(X_train)
X_test = st.transform(X_test)


# In[24]:


def MLP(opt='nadam'):
    X_raw = Input(shape=(X_train.shape[1],), name='input_raw')

    fc1 = BatchNormalization()(X_raw)
    fc1 = Dense(512)(fc1)
    fc1 = PReLU()(fc1)
    fc1 = Dropout(0.25)(fc1)

    fc1 = BatchNormalization()(fc1)
    fc1 = Dense(256)(fc1)
    fc1 = PReLU()(fc1)
    fc1 = Dropout(0.15)(fc1)

    fc1 = BatchNormalization()(fc1)
    auxiliary_output_dense = Dense(1, activation='sigmoid', name='aux_output_dense')(fc1)

    output_all = Dense(1, activation='sigmoid', name='output')(fc1)
    model = Model(input=X_raw, output=output_all)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy')
    return model


# In[25]:


model_mlp=MLP()
#model_name = 'mlp.hdf5'
#model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True,mode='min')
model_mlp.fit(X_train,y,batch_size=1024,nb_epoch=14,verbose=2,
   validation_data=[X_train,y],shuffle=True)
res=model_mlp.predict(X_test,batch_size=1024)
print(np.mean(res))


# In[9]:


early_stopping = EarlyStopping(monitor='val_loss', patience=3)


# In[11]:



model_mlp=MLP()
#model_name = 'mlp.hdf5'
#model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True,mode='min')
model_mlp.fit(X_train,y,batch_size=1024,nb_epoch=25,verbose=2,
   validation_data=[X_test,y_test],shuffle=True,callbacks=[early_stopping])
res=model_mlp.predict(X_test,batch_size=1024)
print(np.mean(res))


# In[21]:


pd.to_pickle(res,'./res/mlp_submission_6_28_offline.pkl')


# In[19]:


result = pd.read_csv('../result/demo_result.csv',index_col=['instanceID'])
result['prob'] = res


# In[ ]:


# result['prob'] = result['prob'].apply(adj)
result.to_csv('./res/mlp_final_sub.csv')


# In[22]:


res.shape


# In[57]:


res=0
best_it=10
fold=1
skf = KFold(n_splits=5, shuffle=True, random_state=1123).split(y)


# In[8]:


def MLP(opt='nadam'):
    X_raw = Input(shape=(X_train.shape[1],), name='input_raw')

    fc1 = BatchNormalization()(X_raw)
    fc1 = Dense(512)(fc1)
    fc1 = PReLU()(fc1)
    fc1 = Dropout(0.25)(fc1)

    fc1 = BatchNormalization()(fc1)
    fc1 = Dense(256)(fc1)
    fc1 = PReLU()(fc1)
    fc1 = Dropout(0.15)(fc1)

    fc1 = BatchNormalization()(fc1)
    auxiliary_output_dense = Dense(1, activation='sigmoid', name='aux_output_dense')(fc1)

    output_all = Dense(1, activation='sigmoid', name='output')(fc1)
    model = Model(input=X_raw, output=output_all)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy')
    return model


# In[10]:


model_mlp=MLP()
#model_name = 'mlp.hdf5'
#model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True,mode='min')
model_mlp.fit(X_ot_train,y_train,batch_size=1024,nb_epoch=2,verbose=2,
   validation_data=[X_ot_test,y_test],shuffle=True)
res=model_mlp.predict(X_test,batch_size=1024)
print(np.mean(tmp_res))


# In[ ]:


result = pd.read_csv('../result/demo_result.csv',index_col=['instanceID'])
result['prob'] = res
# result['prob'] = result['prob'].apply(adj)
result.to_csv('./res/mlp_submission.csv')


# In[58]:


res = 0
for ind_tr, ind_te in skf:
    X_ot_train = X_train[ind_tr]
    X_ot_test=X_train[ind_te]
    y_train = y[ind_tr]
    y_test = y[ind_te]

    model_mlp=MLP()
    #model_name = 'mlp.hdf5'
    #model_checkpoint = ModelCheckpoint(path+model_name, monitor='val_loss', save_best_only=True,mode='min')
    model_mlp.fit(X_ot_train,y_train,batch_size=1024,nb_epoch=2,verbose=2,
       validation_data=[X_ot_test,y_test],shuffle=True)
    tmp_res=model_mlp.predict(X_test,batch_size=1024)
    print(np.mean(tmp_res))
    res+=tmp_res
    print('end fold:{}'.format(fold))
    fold+=1


# In[59]:


print('end bagging')
res=res/5.0
pd.DataFrame(res).describe()


# In[61]:


result = pd.read_csv('../result/demo_result.csv',index_col=['instanceID'])
result['prob'] = res
# result['prob'] = result['prob'].apply(adj)
result.to_csv('../result/submission_mlp.csv')

