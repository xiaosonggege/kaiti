#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: peak_recognition
@time: 2021/6/14 9:45 下午
'''
import numpy as np
import sklearn
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn import manifold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

LEN_PER_FEATURE = 10
class Regression:

    @staticmethod
    def shuffle(data):
        '''
        打乱数据
        :param data: 待处理数据
        :return: 随机打乱后的数据
        '''
        data_shuffle = data
        np.random.shuffle(data_shuffle)
        return data_shuffle

    @classmethod
    def SVR(cls, kernel, C, tol, degree, coef0):
        '''
        多分类SVM分类器
        :param kernel: 选择的核函数 ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        :param C: # 结构风险和经验风险之间的权衡
        # 选择一对多（自己是一类，其他事一类，k类有k个svm）还是一对一（自己是一类，另外一个是一类，k类有k（k-1）/2个svm）
        :param tol: # 停止训练时的误差阈值
        :param degree: 该参数只对多项式核函数有用，是指多项式核函数的阶数n
        :param coef0: 核函数中的独立项，只对多项式核函数和sigmod核函数有用，是指其中的参数C
        :return: SVM对象
        '''
        svr = SVR(
            kernel= kernel,
            C= C,
            tol= tol,
            gamma='auto',
            degree= degree,
            coef0= coef0
        )

        return svr

    @classmethod
    def Adaboost(cls, max_depth, min_samples_split, min_samples_leaf, algorithm, n_estimators, learning_rate):
        '''
        多分类CART树
        :param max_depth: 树最大深度
        :param min_samples_split: 继续划分叶子结点所需要的最小例子数
        :param min_samples_leaf: 叶子结点中最少要有的实例数量
        :param algorithm:  If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities. If 'SAMME'
         then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically
         converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
        :param n_estimators: 集成子学习器的最大数量
        :param learning_rate: 控制每类数据的分布的收缩
        :return: Adaboost对象
        '''
        #构建CART决策树作为子学习器
        clf = DecisionTreeRegressor(
            max_depth= max_depth,
            min_samples_split= min_samples_split,
            min_samples_leaf= min_samples_leaf
        )

        #构建Adaboost对象
        bdt = AdaBoostRegressor(
            base_estimator= clf,
            # algorithm= algorithm,
            n_estimators= n_estimators,
            learning_rate= learning_rate
        )

        return bdt

    @classmethod
    def XGBoost(cls, max_depth, learning_rate, n_estimators, objective, nthread, gamma, min_child_weight,
                      subsample, reg_lambda, scale_pos_weight):
        '''
        XGBoost对象
        :param max_depth: 树的最大深度
        :param learning_rate: 学习率
        :param n_estimators: 树的个数
        :param objective: 损失函数类型
       'reg:logistic' –逻辑回归。
       'binary:logistic' –二分类的逻辑回归问题，输出为概率。
       'binary:logitraw' –二分类的逻辑回归问题，输出的结果为wTx。
       'count:poisson' –计数问题的poisson回归，输出结果为poisson分布。在poisson回归中，max_delta_step的缺省值为0.7。(used to safeguard optimization)
       'multi:softmax' –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
       'multi:softprob' –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。没行数据表示样本所属于每个类别的概率。
       'rank:pairwise' –set XGBoost to do ranking task by minimizing the pairwise loss
        :param nthread: 线程数
        :param gamma: 节点分裂时损失函数所需最小下降值
        :param min_child_weight: 叶子结点最小权重
        :param subsample: 随机选择样本比例建立决策树
        :param reg_lambda: 二阶范数正则化项权衡值
        :param scale_pos_weight: 解决样本个数不平衡问题
        :return: XGBoost对象
        '''
        xgbc = XGBRegressor(
            max_depth= max_depth,
            learning_rate= learning_rate,
            n_estimators= n_estimators,
            objective= objective,
            nthread= nthread,
            gamma= gamma,
            min_child_weight= min_child_weight,
            subsample= subsample,
            colsample_bytree= subsample,
            reg_lambda= reg_lambda,
            scale_pos_weight= scale_pos_weight,
            random_state= 32,
        )

        return xgbc

    @classmethod
    def MiniBatchDictionaryLearning(cls):
        '''字典学习'''
        minidictlearn = MiniBatchDictionaryLearning(
            n_components=15, transform_algorithm='lasso_lars', random_state=42,
        )
        return minidictlearn

    def __init__(self, dataset_findpeak:np.ndarray, dataset_distinct:np.ndarray):
        '''
        分类器构造函数
        :param dataset:
        '''
        self._dataset_findpeak = self.shuffle(dataset_findpeak)
        self._dataset_distinct = self.shuffle(dataset_distinct)

    def training_main_findpeak(self, model_name, model, save_path, Threshold=None):
        '''
        本地甄别针对多个模型进行训练操作
        :param model_name: 模型名称
        :param model: 需要训练的模型
        :return: None
        '''
        #初始化k折平均查准率，k折平均查全率，k折平均F1参数
        precision_rate, recall_rate, F1_rate = 0, 0, 0
        # k-fold对象,用于生成训练集和交叉验证集数据
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32)
        # 交叉验证次数序号
        fold = 1

        pred_cv = None
        for train_data_index, cv_data_index in kf.split(self._dataset_findpeak):
            # 找到对应索引数据
            train_data, cv_data = self._dataset_findpeak[train_data_index], self._dataset_findpeak[cv_data_index]
            # print(np.isnan(train_data).any(), np.isnan(cv_data).any())
            # print(train_data.shape, cv_data.shape)
            # 训练数据
            # print(train_data[:, :train_data.shape[-1]//2].shape, train_data[:, train_data.shape[-1]//2:].shape)
            label_i = train_data.shape[-1] // 2
            for each_model in model:
                each_model.fit(X=train_data[:, :train_data.shape[-1]//2], y=train_data[:, label_i])
                label_i += 1

            # print('第%s折模型训练集精度为: %s' % (fold, model.score(train_data[:, :train_data.shape[0]//2], train_data[:, train_data.shape[0]//2:])))

            # 对验证集进行预测
            pred_cv_sub = None
            if isinstance(model, MiniBatchDictionaryLearning):
                pred_cv = model.transform(cv_data[:, :train_data.shape[-1]//2])
            else:
                for each_model in model:
                    # pred_cv_sub = np.append(pred_cv_sub, each_model.predict(cv_data[:, :train_data.shape[-1]//2]))
                    # print(each_model.predict(cv_data[:, :train_data.shape[-1]//2])[:, np.newaxis].shape)
                    pred_cv_sub = each_model.predict(cv_data[:, :train_data.shape[-1]//2])[:, np.newaxis] if pred_cv_sub is None \
                    else np.hstack((pred_cv_sub, each_model.predict(cv_data[:, :train_data.shape[-1]//2])[:, np.newaxis]))
            # 对验证数据进行指标评估
            # precision_rate = precision_rate *
            fold += 1
            pred_cv = pred_cv_sub if pred_cv is None else np.vstack((pred_cv, pred_cv_sub))

        print(pred_cv, pred_cv.shape)

    def training_main_distinct(self, model_name, model, save_path, Threshold=None):
        '''
        针对多个模型进行训练操作
        :param model_name: 模型名称
        :param model: 需要训练的模型
        :return: None
        '''

        # 初始化k折平均查准率，k折平均查全率，k折平均F1参数
        precision_rate, recall_rate, F1_rate = 0, 0, 0
        # k-fold对象,用于生成训练集和交叉验证集数据
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32)
        # 交叉验证次数序号
        fold = 1

        pred_cv = None
        for train_data_index, cv_data_index in kf.split(self._dataset_findpeak):
            # 找到对应索引数据
            train_data, cv_data = self._dataset_findpeak[train_data_index], self._dataset_findpeak[cv_data_index]
            # print(np.isnan(train_data).any(), np.isnan(cv_data).any())
            # print(train_data.shape, cv_data.shape)
            # 训练数据
            # print(train_data[:, :train_data.shape[-1]//2].shape, train_data[:, train_data.shape[-1]//2:].shape)
            label_i = train_data.shape[-1] // 2
            for each_model in model:
                each_model.fit(X=train_data[:, :train_data.shape[-1] // 2], y=train_data[:, label_i])
                label_i += 1

            # print('第%s折模型训练集精度为: %s' % (fold, model.score(train_data[:, :train_data.shape[0]//2], train_data[:, train_data.shape[0]//2:])))

            # 对验证集进行预测
            pred_cv_sub = None
            if isinstance(model, MiniBatchDictionaryLearning):
                pred_cv = model.transform(cv_data[:, :train_data.shape[-1] // 2])
            else:
                for each_model in model:
                    # pred_cv_sub = np.append(pred_cv_sub, each_model.predict(cv_data[:, :train_data.shape[-1]//2]))
                    # print(each_model.predict(cv_data[:, :train_data.shape[-1]//2])[:, np.newaxis].shape)
                    pred_cv_sub = each_model.predict(cv_data[:, :train_data.shape[-1] // 2])[:,
                                  np.newaxis] if pred_cv_sub is None \
                        else np.hstack(
                        (pred_cv_sub, each_model.predict(cv_data[:, :train_data.shape[-1] // 2])[:, np.newaxis]))
            # 对验证数据进行指标评估
            # precision_rate = precision_rate *
            fold += 1
            pred_cv = pred_cv_sub if pred_cv is None else np.vstack((pred_cv, pred_cv_sub))

        print(pred_cv, pred_cv.shape)

def model_main(dataset_findpeak, dataset_distinct, operation):
    '''
    训练主函数
    :param dataset: 训练集数据全部
    :param operation: 选择训练的模型, 'SVR', 'Adaboost', 'XGBoost'
    :return: None
    '''
    regression = Regression(dataset_findpeak=dataset_findpeak, dataset_distinct=dataset_distinct)
    if operation == 'SVR':
        #SVM分类器训练
        SVR = [Regression.SVR(kernel= 'rbf', C= 1.0, tol= 1e-3, degree= 3, coef0= 1) for _ in range(LEN_PER_FEATURE)]
        regression.training_main_findpeak(model_name= 'SVM分类器', model=SVR, save_path='./svm_model_findpeak')
        regression.training_main_distinct(model_name= 'SVM分类器', model=SVR, save_path='./svm_model_distinct')
    elif operation == 'Adaboost':
        #Adaboost分类器训练
        Adaboost = [Regression.Adaboost(max_depth=2, min_samples_split=2, min_samples_leaf=1,
                                                   algorithm='SAMME.R', n_estimators=500, learning_rate=1e-2)\
                    for _ in range(LEN_PER_FEATURE)]
        regression.training_main_findpeak(model_name='Adaboost分类器', model=Adaboost, save_path='./adaboost_model_findpeak')
        regression.training_main_distinct(model_name='Adaboost分类器', model=Adaboost, save_path='./adaboost_model_distinct')


    elif operation == 'XGBoost':
        #XGBoost分类器训练
        XGBoost = [Regression.XGBoost(max_depth=2, learning_rate=1e-2, n_estimators=200,
                                                 objective='reg:squarederror', nthread=4, gamma=0.1,
                                                 min_child_weight=1, subsample=1, reg_lambda=2, scale_pos_weight=1.)\
                   for _ in range(LEN_PER_FEATURE)]
        regression.training_main_findpeak(model_name='XGBoost分类器', model=XGBoost, save_path='./xgboost_model_findpeak')
        regression.training_main_distinct(model_name='XGBoost分类器', model=XGBoost, save_path='./xgboost_model_distinct')
        for each_xgb in XGBoost:
            digraph = xgb.to_graphviz(each_xgb, num_trees=2)
            digraph.format = 'png'
            digraph.view('./signal_xgb')
            xgb.plot_importance(each_xgb)
            plt.show()

    elif operation == 'minidictlearn':
        #XGBoost分类器训练
        minidictlearn = [Regression.MiniBatchDictionaryLearning()]
        regression.training_main_findpeak(model_name='minidictlearn分类器', model=minidictlearn, save_path='./minidictlearn_model_findpeak')
        regression.training_main_distinct(model_name='minidictlearn分类器', model=minidictlearn, save_path='./minidictlearn_model_distinct')

if __name__ == '__main__':
    # ========>debug========>
    #findpeak_debug
    dataset_findpeak = np.random.normal(size=(100, 10+10))
    print(dataset_findpeak.shape)
    #distinct_debug
    dataset_distinct = np.random.normal(size=(100, 10+10))

    model_main(dataset_findpeak=dataset_findpeak, dataset_distinct=dataset_distinct, operation='XGBoost')


