# coding=utf-8

# @Time    : 2019-01-12 10:59
# @Auther  : Batista-yu
# @Contact : 550461053@gmail.com
# @license : (C) Copyright2016-2018, Batista Yu Limited.

'''
采用Auto ml 
'''

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)



import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='64G')



# data path
pre_root_path = "data/pre-data"
result_path = "result"
train = pd.read_csv(pre_root_path + '/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv(pre_root_path + '/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
print('load data')

# train = h2o.upload_file(pre_root_path + '/jinnan_round1_train_20181227.csv')#"./cache_data/train_data_f{}.csv".format(1))
# test = h2o.upload_file(pre_root_path + '/jinnan_round1_testA_20181227.csv')#"./cache_data/test_data_f{}.csv".format(1))


############# 特征工程 start

# 删除类别唯一的特征
for df in [train, test]:
    df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)

# 删除某一类别占比超过90%的列
good_cols = list(train.columns)
for col in train.columns:
    rate = train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.9:
        good_cols.remove(col)
        print(col, rate)

# 暂时不删除，后面构造特征需要
good_cols.append('A1')
good_cols.append('A3')
good_cols.append('A4')

good_cols.append('A2')

# 删除异常值
train = train[train['收率'] > 0.87]  # 0.87以下的很少，几个，但是也不能作为是异常值！

# show = train[train['收率'] >= 1.00]
# print('收率为1：')
# print(show)
# input()

train = train[good_cols]
good_cols.remove('收率')
test = test[good_cols]

# 合并数据集
target = train['收率']
del train['收率']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)


def timeTranSecond(t):
    try:
        t, m, s = t.split(":")
    except:
        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            return 0

    try:
        tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
    except:
        return (30 * 60) / 3600

    return tm




for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f, '应该在前面被删除了！')

def  trans_temper(x):
    if not x:
        return 0
    return x + 273.15
    # return x * 9 / 5.0 + 32.0

for f in ['A6', 'A8', 'A10', 'A12', 'A15', 'A17', 'A21', 'A25', 'A27', 'B6', 'B8']:
    try:
        data[f] = data[f].apply(trans_temper)
    except:
        print(f, '出错了')


def getDuration(se):
    try:
        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh) > int(eh):
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
        else:
            tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
    except:
        if se == '19:-20:05':
            return 1
        elif se == '15:00-1600':
            return 1

    return tm


for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)


data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

categorical_columns = [f for f in data.columns if f not in ['样本id']]
numerical_columns = [f for f in data.columns if f not in categorical_columns]


# 有风的冬老哥，在群里无意爆出来的特征，让我提升了三个个点，当然也可以顺此继续扩展
data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])

numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')

del data['A1']
del data['A2']  # A2+A3+A4是定值
del data['A3']
del data['A4']
# del data['A9']
categorical_columns.remove('A1')
categorical_columns.remove('A2')  # A2+A3+A4是定值
categorical_columns.remove('A3')
categorical_columns.remove('A4')
# categorical_columns.remove('A9')

#label encoder
for f in categorical_columns:
    data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
train = data[:train.shape[0]]
test  = data[train.shape[0]:]
print(train.shape)
print(test.shape)


# train['target'] = list(target)
train['target'] = target
train['intTarget'] = pd.cut(train['target'], 5, labels=False)
train = pd.get_dummies(train, columns=['intTarget'])
li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
mean_columns = []
for f1 in categorical_columns:
    cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
    if cate_rate < 0.90:
        for f2 in li:
            col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
            mean_columns.append(col_name)
            order_label = train.groupby([f1])[f2].mean()
            train[col_name] = train['B14'].map(order_label)
            miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
            if miss_rate > 0:
                train = train.drop([col_name], axis=1)
                mean_columns.remove(col_name)
            else:
                test[col_name] = test['B14'].map(order_label)

train.drop(li + ['target'], axis=1, inplace=True)
print(train.shape)
print(test.shape)


X_train = train[mean_columns+numerical_columns].values
X_test = test[mean_columns+numerical_columns].values
# one hot
enc = OneHotEncoder()
for f in categorical_columns:
    enc.fit(data[f].values.reshape(-1, 1))
    X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
    X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
print(X_train.shape)
print(X_test.shape)


y_train = target.values



############# 特征工程 end

# train = h2o.upload_file(pre_root_path + '/jinnan_round1_train_20181227.csv')#"./cache_data/train_data_f{}.csv".format(1))
# test = h2o.upload_file(pre_root_path + '/jinnan_round1_testA_20181227.csv')#"./cache_data/test_data_f{}.csv".format(1))
print('load data to h2o')
train = h2o.load_dataset(X_train)
test = h2o.load_dataset(X_test)

# all_data = h2o.connect(train, test)
# all_data = X_train + X_test

# all_data = ' '  # 这是所有的特征，可以从参考baseline1
# feature_name = [i for i in all_data.columns if i not in ['样本id','收率']]
feature_name = mean_columns+numerical_columns
x = feature_name
y = '收率'

aml = H2OAutoML(max_models=320, seed=2019, max_runtime_secs=12800)
aml.train(x=feature_name, y=y, training_frame=train)

lb = aml.leaderboard
lb.head(rows=lb.nrows)

automl_predictions = aml.predict(test).as_data_frame().values.flatten()
print('predict:')
print('the automl predictions is:', automl_predictions)

# 保存
sub_df = pd.read_csv(pre_root_path + '/jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = automl_predictions
# sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))  # 这是覆盖读取的文件
sub_df.to_csv(result_path + '/jinnan_round1_submit_20181227_2.csv', index=0, header=0)  # 这是另存为，不保存索引行
print('save done!')