#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
"""


import pandas as pd


ffm = pd.read_csv('../result/submission_ffm.csv')
lgb = pd.read_csv('../result/submission_lgb.csv')# lgb 4 5 + hhy:lgb 4   lgb_5*0.6 + lgb_4_hhy*0.3 +lgb_4*0.1
mlp = pd.read_csv('../result/submission_mlp.csv')
xgb = pd.read_csv('../result/submission_xgb.csv')#xgb_5*0.6 + xgb_4*0.4

result = pd.read_csv('../result/demo_result.csv')
result['prob'] = lgb['prob']*0.4 + xgb['prob']*0.4 + ffm['prob']*0.15 + mlp['prob']*0.05


result.to_csv('../result/submission_final.csv',)

