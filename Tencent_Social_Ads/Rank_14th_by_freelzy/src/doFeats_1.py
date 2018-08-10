#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: freelzy
@Github: https://github.com/freelzy
"""

import numpy as np
import pandas as pd
import scipy as sp
import gc
import datetime
import random
import scipy.special as special

rawpath='C:/final/'
temppath='C:/final/temp/'
iapath='C:/final/temp/installedactions/'

class HyperParam(object):#平滑，这个快一点；hyper=HyperParam(1, 1); hyper.update_from_data_by_moment(show, click)
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)
     
class BayesianSmoothing(object):#贝叶斯平滑，这个慢一点
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
            print(self.alpha, self.beta)

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)

def readData(m_type='inner',drop=True):  ###################使用Trick时，left merge不改变顺序会比inner merge差1个万分点左右？
    X_train = pd.read_csv(rawpath+'train.csv')
    X_test = pd.read_csv(rawpath+'test.csv')
    if drop:
        X_train.drop('conversionTime', axis=1, inplace=True)

    userfile = pd.read_csv(rawpath+'user.csv')
    X_train = X_train.merge(userfile, how=m_type, on='userID')
    X_test = X_test.merge(userfile, how=m_type, on='userID')
    del userfile
    gc.collect()

    adfile = pd.read_csv(rawpath+'ad.csv')
    X_train = X_train.merge(adfile, how=m_type, on='creativeID')
    X_test = X_test.merge(adfile, how=m_type, on='creativeID')
    del adfile
    gc.collect()

    appcatfile = pd.read_csv(rawpath+'app_categories.csv')
    X_train = X_train.merge(appcatfile, how=m_type, on='appID')
    X_test = X_test.merge(appcatfile, how=m_type, on='appID')
    del appcatfile
    gc.collect()

    positionfile = pd.read_csv(rawpath+'position.csv')
    X_train = X_train.merge(positionfile, how=m_type, on='positionID')
    X_test = X_test.merge(positionfile, how=m_type, on='positionID')
    del positionfile
    gc.collect()
    print('merge type:', m_type)
    return X_train, X_test

def logloss(act, preds):
    epsilon = 1e-15
    preds = sp.maximum(epsilon, preds)
    preds = sp.minimum(1 - epsilon, preds)
    ll = sum(act * sp.log(preds) + sp.subtract(1, act) * sp.log(sp.subtract(1, preds)))
    ll = ll * -1.0 / len(act)
    return ll

# 添加时间特征：天级 + 小时级
def doPre(data):
    data['day'] = data['clickTime'] // 1000000
    data['hour'] = data['clickTime'] % 1000000 // 10000
    return data

# installed文件关联user和app文件提取特征：和20th的玩法一样：使用left_join即可
userfile= pd.read_csv(rawpath+'user.csv')
appfile= pd.read_csv(rawpath+'app_categories.csv')
installed = pd.read_csv(rawpath+'user_installedapps.csv')
installed=installed.merge(userfile,how='left',on='userID')
installed=installed.merge(appfile,how='left',on='appID')

#app，appCat安装用户数
"""
 这里是一个trick：
- groupby的计算，只用单层中括号，且进行reset_index()
- 接着调用.columns=['a','b']，'a'一定是groupby计算所使用的key
"""
temp = installed.groupby('appID')['userID'].count().reset_index()
temp.columns=['appID','app_usercount'] 
temp.to_csv(iapath+'appInstalledusercount.csv',index=False)
temp = installed.groupby('appCategory')['userID'].count().reset_index()
temp.columns=['appCategory','appCat_usercount']
temp.to_csv(iapath+'appCatInstalledusercount.csv',index=False)

#user，edu，age,gender安装app数
temp = installed.groupby('userID')['appID'].count().reset_index()
temp.columns=['userID','user_appcount']
temp.to_csv(iapath+'userInstalledappscount.csv',index=False)
temp = installed.groupby('education')['appID'].count().reset_index()
temp.columns=['education','edu_appcount']
temp.to_csv(iapath+'eduuserInstalledappscount.csv',index=False)
temp = installed.groupby('age')['appID'].count().reset_index()
temp.columns=['age','age_appcount']
temp.to_csv(iapath+'ageuserInstalledappscount.csv',index=False)
temp = installed.groupby('gender')['appID'].count().reset_index()
temp.columns=['gender','gender_appcount']
temp.to_csv(iapath+'genderuserInstalledappscount.csv',index=False)
print('installed over...')

# actions文件提取特征，7天滑窗，统计用户安装的app数，app被安装的用户数  --- 很不错的一个构建特征的方法
"""
新学trick:
- 使用滑动时间窗口来统计特征的变化 【分别从appID和userID两个维度】
- 代码借鉴：for loop
"""
actions = pd.read_csv(rawpath+'user_app_actions.csv')
actions['day']=actions['installTime']//1000000
res=pd.DataFrame()
temp=actions[['userID','day','appID']]
for day in range(28,32):
    count=temp.groupby(['userID']).apply(lambda x: x['appID'][(x['day']<day).values & (x['day']>day-8).values].count()).reset_index(name='appcount')
    count['day']=day
    res=res.append(count,ignore_index=True)
res.to_csv(iapath+'all_user_seven_day_cnt.csv',index=False)
res=pd.DataFrame()
temp=actions[['userID','day','appID']]
for day in range(28,32):
    count=temp.groupby(['appID']).apply(lambda x: x['userID'][(x['day']<day).values & (x['day']>day-8).values].count()).reset_index(name='usercount')
    count['day']=day
    res=res.append(count,ignore_index=True)
res.to_csv(iapath+'all_app_seven_day_cnt.csv',index=False)
print('actions over...')


X_loc_train,X_loc_test=readData(m_type='inner',drop=True)
print('readData over')
X_loc_train=doPre(X_loc_train)
X_loc_test=doPre(X_loc_test)
print('doPre over...')


# 初赛用7天滑窗算统计，决赛根据周冠军分享改为了使用了clickTime之前所有天算统计
"""
新学trick:
- ctr和cvr 特征的计算，使用多天时间窗口的平滑来计算，maybe可以一定程度上解决延时数据返回的问题
- ctr和cvr 特征的计算，除了使用一阶特征外，还可以用二阶特征、三阶特征等。
- 使用哪些低阶特征和高阶特征进行统计，取决于经验和实验
- 高阶特征可以用元组数据格式进行存储，便于循环时的调用

PS:
- 计算转化率的本质：
    - clickTran(data)使用小时级的时间划窗来计算（bettenW大神也使用过小时级的时间划窗来计算）
    - new_clickTran(data) 使用天级的时间划窗来计算
    - 实验证明，天级的更科学，因为普遍存在延迟返回。
"""

# 单特征的转化率特征计算
first_order_feature = ['creativeID','positionID','userID','sitesetID']
for feat_1 in first_order_feature:  
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,'day','label']]
    # 统计每一天的ctr变化
    for day in range(28,32):
        # 计算展示数
        count=temp.groupby([feat_1]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_all')
        # 计算点击数
        count1=temp.groupby([feat_1]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_1')
        # 二者合并
        count[feat_1+'_1']=count1[feat_1+'_1'] 
        # 有些广告可能完全没有转化：没有任何点击，So需要进行0填充
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
    res.to_csv(temppath+'%s.csv' %feat_1, index=False)

# 二阶特征的转化率特征计算
second_order_feature = [('positionID','advertiserID'),('userID','sitesetID'),('positionID','connectionType'),('userID','positionID'),
                        ('appPlatform','positionType'),('advertiserID','connectionType'),('positionID','appCategory'),('appID','age'),
                        ('userID', 'appID'),('userID','connectionType'),('appCategory','connectionType'),('appID','hour'),('hour','age')]
for feat_1,feat_2 in second_order_feature:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,feat_2,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_all')
        count1=temp.groupby([feat_1,feat_2]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_1')
        count[feat_1+'_'+feat_2+'_1']=count1[feat_1+'_'+feat_2+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,' over')
    res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2), index=False)

# 三阶特征的转化率特征计算
third_order_feature = [('appID','connectionType','positionID'),('appID','haveBaby','gender')]
for feat_1,feat_2,feat_3 in third_order_feature:
    gc.collect()
    res=pd.DataFrame()
    temp=X_loc_train[[feat_1,feat_2,feat_3,'day','label']]
    for day in range(28,32):
        count=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['day']<day).values].count()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_all')
        count1=temp.groupby([feat_1,feat_2,feat_3]).apply(lambda x: x['label'][(x['day']<day).values].sum()).reset_index(name=feat_1+'_'+feat_2+'_'+feat_3+'_1')
        count[feat_1+'_'+feat_2+'_'+feat_3+'_1']=count1[feat_1+'_'+feat_2+'_'+feat_3+'_1']
        count.fillna(value=0, inplace=True)
        count['day']=day
        res=res.append(count,ignore_index=True)
    print(feat_1,feat_2,feat_3,' over')
    res.to_csv(temppath+'%s.csv' % (feat_1+'_'+feat_2+'_'+feat_3), index=False)


# 比赛官方群里大神分享过的，这里用app平均回流时间做特征，缺失的用app类别的平均回流时间替代【只有merge_type是left_join的时候才会有缺失】
"""
新学Trick：转化时间的快慢也是一个重要的特征
    - 不同的广告主会有不同的转化设定；
    - 这里只考虑转化时间的长短，粒度还是有些粗；
    - 这一部分，个人感觉还是20th的团队做的更细致；
"""

X_loc_train,X_loc_test=readData(m_type='inner',drop=False)
del X_loc_test
X_loc_train=X_loc_train.loc[X_loc_train['label']==1,:]
X_loc_train['cov_diffTime']=X_loc_train['conversionTime']-X_loc_train['clickTime']
grouped=X_loc_train.groupby('appID')['cov_diffTime'].mean().reset_index()
grouped.columns=['appID','cov_diffTime']
grouped.to_csv(temppath+'app_cov_diffTime.csv',index=False)

grouped=X_loc_train.groupby('appCategory')['cov_diffTime'].mean().reset_index()
grouped.columns=['appCategory','appCat_cov_diffTime']
grouped.to_csv(temppath+'appCat_cov_diffTime.csv',index=False)


