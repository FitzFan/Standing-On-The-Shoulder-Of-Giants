#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Ad_Utils.py
@time: 2018/3/7 16:27
@desc: 
    - 本脚本是为训练ffm服务
    - 因为ffm不是自己写，是调用他人的接口，所以对数据形式有一定的要求
    - 故此脚本是专门产出符合标准的数据
"""

import hashlib,time,sys
import numpy as np
from tqdm import tqdm
NR_BINS = 100000

def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)
    
def gen_hash_row(feats,label):
    result = []
    for idx, item in enumerate(feats):
        val = item.split('-')[-1]
        if val != 'nan':
                result.append(gen_hash_item(idx,item))
    lbl = 1
    if label == 0:
        lbl = 0
    return str(lbl) + ' ' + ' '.join(result)+'\n'
    
map_col = lambda dat,col: col+"-"+dat.map(str)
gen_hash_item = lambda field, feat: '{0}:{1}:1'.format(field,hashstr(feat))

def data2libffm(merge_dat,output_name):
    start = time.time()
    merge_dat_val = merge_dat.drop(['label'],axis=1)
    cols = merge_dat_val.columns
    features = []
    for col in merge_dat_val.columns:
        features.append(map_col(merge_dat_val[col],col))
    features = np.array(features).T
    with open(output_name,'w') as f_tr:
        i = 0;
        for item,label in tqdm(zip(features,merge_dat['label'])):
            if(i%1000000==0):
                sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))
            row = gen_hash_row(item,label)
            f_tr.write(row)
            i+=1
    print('finish convert data to libffm')
