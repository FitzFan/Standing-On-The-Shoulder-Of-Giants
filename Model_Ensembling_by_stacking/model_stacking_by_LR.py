#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Emanuele Olivetti
@Github: https://github.com/emanuele
@Desc  : - 用logistic regression做stacking
         - 该script中，Emanuele中将stacking称为blending
         - 有大神用这个脚本，Stacking 8 base models (diverse ET’s, RF’s and GBM’s) with Logistic Regression gave me my second best score of 0.99409 accuracy, good for first place.
"""

from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':
    np.random.seed(0)  # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    X, y, X_submission = load_data.load() # X_submission <=> X_test

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            """
            这里的X_train, y_train, X_test, y_test均是有label的total_train_set的子集
            """
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1] # 模型对y_test的预测
            dataset_blend_train[test, j] = y_submission    # 当cv_fold全部结束后，就完成了当前model对 X 的全部预测。若不懂就谷歌一下StratifiedKFold()
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1] # 对真实测试集进行预测，有多少cv_fold，有预测多少次
        # mean(1)是按列求平均, mean(0)是按行求平均，做一下实验就懂了
        # 对同一个model，不同cv的预测结果求平均，**也有大牛认为其实不用做CV，全量跑一次更好！
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    """
    loop结束的output:
        - 所有模型对全体训练集的预测，记为dataset_blend_train，有多少个model就有多少列，每一列都视为feature
        - 所有模型对全体测试集的预测的平均值，记为dataset_blend_test
    """

    print "Blending."
    clf = LogisticRegression()
    """
    - LR来做stacking：用LR来fit每一个模型的weight。
    - 作者在这里将stacking和blending视为同一个东西，个人认为还是有些区别
    """
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print "Saving Results."
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')

    """
    逻辑梳理：
        - 使用StratifiedKFold()切分数据集
        - 定义N个不同的model
        - 两层循环：
            - 外层：model_loop
            - 内存：cv_loop
            - 输出：
                - 所有模型对全体训练集的预测，记为dataset_blend_train，有多少个model就有多少列，每一列都视为feature
                - 所有模型对全体测试集的预测的平均值，记为dataset_blend_test
        - LR来做stacking：用LR来fit每一个模型的weight
    """


