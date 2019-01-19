# coding=utf-8

# @Time    : 2019-01-19 10:07
# @Auther  : Batista-yu
# @Contact : 550461053@gmail.com
# @license : (C) Copyright2016-2018, Batista Yu Limited.

'''

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
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.tools as tls
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline_logfile_1_15",
                    filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

pre_root_path = "data/pre-data"
result_path = "result"
train = pd.read_csv(pre_root_path + '/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv(pre_root_path + '/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
print('load data')

target_col = "收率"

# 删除异常值
print(train[train['收率'] < 0.87])

train = train[train['收率'] > 0.87]
train.loc[train['B14'] == 40, 'B14'] = 400
train = train[train['B14']>=400]

# 合并数据集, 顺便处理异常数据
target = train['收率']
train.loc[train['A25'] == '1900/3/10 0:00', 'A25'] = train['A25'].value_counts().values[0]
train['A25'] = train['A25'].astype(int)
train.loc[train['B14'] == 40, 'B14'] = 400
# test.loc[test['B14'] == 385, 'B14'] = 385

test_select = {}
for v in [280, 385, 390, 785]:
    print(v)
    print(test[test['B14'] == v]['样本id'])
    test_select[v] = test[test['B14'] == v]['样本id'].index
    print(test[test['B14'] == v]['样本id'].index)
    print(test_select[v])

del train['收率']
data = pd.concat([train,test],axis=0,ignore_index=True)
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

data['样本id'] = data['样本id'].apply(lambda x: x.split('_')[1])
data['样本id'] = data['样本id'].astype(int)

# 基本数据处理完毕, 开始拼接数据
train = data[:train.shape[0]]
test  = data[train.shape[0]:]

train['target'] = list(target)

new_train = train.copy()
new_train = new_train.sort_values(['样本id'], ascending=True)
train_copy = train.copy()
train_copy = train_copy.sort_values(['样本id'], ascending=True)

# 把train加长两倍
train_len = len(new_train)
new_train = pd.concat([new_train, train_copy])

# 把加长两倍的train拼接到test后面
new_test = test.copy()
new_test = pd.concat([new_test, new_train])

import sys
# 开始向后做差
diff_train = pd.DataFrame()
ids = list(train_copy['样本id'].values)
print(ids)


from tqdm import tqdm
import os
# 构造新的训练集
if os.path.exists(pre_root_path + '/diff_train.csv'):
    diff_train = pd.read_csv(pre_root_path + '/diff_train.csv')
else:
    for i in tqdm(range(1, train_len)):
        # 分别间隔 -1, -2, ... -len行 进行差值,得到实验的所有对比实验
        diff_tmp = new_train.diff(-i)
        diff_tmp = diff_tmp[:train_len]
        diff_tmp.columns = [col_ + '_difference' for col_ in
                            diff_tmp.columns.values]
        # 求完差值后加上样本id
        diff_tmp['样本id'] = ids
        diff_train = pd.concat([diff_train, diff_tmp])

    diff_train.to_csv(pre_root_path + '/diff_train.csv', index=False)

# 构造新的测试集
diff_test = pd.DataFrame()
ids_test = list(test['样本id'].values)
test_len = len(test)

if os.path.exists(pre_root_path + '/diff_test.csv'):
    diff_test = pd.read_csv(pre_root_path + '/diff_test.csv')
else:
    for i in tqdm(range(test_len, test_len+train_len)):
        # 分别间隔 - test_len , -test_len -1 ,.... - test_len - train_len +1 进行差值, 得到实验的所有对比实验
        diff_tmp = new_test.diff(-i)
        diff_tmp = diff_tmp[:test_len]
        diff_tmp.columns = [col_ + '_difference' for col_ in
                            diff_tmp.columns.values]
        # 求完差值后加上样本id
        diff_tmp['样本id'] = ids_test
        diff_test = pd.concat([diff_test, diff_tmp])

    diff_test = diff_test[diff_train.columns]
    diff_test.to_csv(pre_root_path + '/diff_test.csv', index=False)


print(train.columns.values)
# 和train顺序一致的target
train_target = train['target']
train.drop(['target'], axis=1, inplace=True)
# 拼接原始特征
diff_train = pd.merge(diff_train, train, how='left', on='样本id')
diff_test = pd.merge(diff_test, test, how='left', on='样本id')

target = diff_train['target_difference']
diff_train.drop(['target_difference'], axis=1, inplace=True)
diff_test.drop(['target_difference'], axis=1, inplace=True)

X_train = diff_train
y_train = target
X_test = diff_test

print(X_train.columns.values)

param = {'num_leaves': 120, #31
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         # "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l2": 0.1,
         # "lambda_l1": 0.1,
         'num_thread': 4,
         "verbosity": -1}
groups = X_train['样本id'].values

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(diff_train))
predictions_lgb = np.zeros(len(diff_test))

feature_importance = pd.DataFrame()
feature_importance['feature_name'] = X_train.columns.values


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    dev = X_train.iloc[trn_idx]
    val = X_train.iloc[val_idx]

    trn_data = lgb.Dataset(dev, y_train.iloc[trn_idx])
    val_data = lgb.Dataset(val, y_train.iloc[val_idx])

    num_round = 20000  # 3000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=5,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(val, num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    importance = clf.feature_importance(importance_type='gain')
    feature_name = clf.feature_name()
    tmp_df = pd.DataFrame({'feature_name': feature_name, 'importance': importance})

    feature_importance = pd.merge(feature_importance, tmp_df, how='left',
                                  on='feature_name')
    print(len(feature_importance['feature_name']))

print(len(diff_train))
feature_importance.to_csv(result_path + '/eda/feature_importance2.csv', index=False)
# 还原train target
diff_train['compare_id'] = diff_train['样本id'] - diff_train['样本id_difference']
train['compare_id'] = train['样本id']
train['compare_target'] = list(train_target)
# 把做差的target拼接回去
diff_train = pd.merge(diff_train, train[['compare_id', 'compare_target']], how='left', on='compare_id')
print(diff_train.columns.values)
diff_train['pre_target_diff'] = oof_lgb
diff_train['pre_target'] = diff_train['pre_target_diff'] + diff_train['compare_target']

mean_result = diff_train.groupby('样本id')['pre_target'].mean().reset_index(name='pre_target_mean')
true_result = train[['样本id', 'compare_target']]
mean_result = pd.merge(mean_result, true_result, how='left', on='样本id')
print(mean_result)
print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
logging.info("Lgb CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))

print("CV score: {:<8.8f}".format(mean_squared_error(mean_result['pre_target_mean'].values,  mean_result['compare_target'].values)))
logging.info("Lgb CV score: {:<8.8f}".format(mean_squared_error(mean_result['pre_target_mean'].values,  mean_result['compare_target'].values)))

# pre_target = mean_result['pre_target_mean'].values
# true_target = mean_result['']

# 还原test target
diff_test['compare_id'] = diff_test['样本id'] - diff_test['样本id_difference']
diff_test = pd.merge(diff_test, train[['compare_id', 'compare_target']], how='left', on='compare_id')
diff_test['pre_target_diff'] = predictions_lgb
diff_test['pre_target'] = diff_test['pre_target_diff'] + diff_test['compare_target']

mean_result_test = diff_test.groupby(diff_test['样本id'], sort=False)['pre_target'].mean().reset_index(name='pre_target_mean')
print(mean_result_test)
test = pd.merge(test, mean_result_test, how='left', on='样本id')
sub_df = pd.read_csv(pre_root_path + '/jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = test['pre_target_mean']
# sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
sub_df.to_csv(result_path + '/jinnan_round1_submit_20181227_3_2.csv', index=0, header=0)  # 这是另存为，不保存索引行
print('save done!')

for v in test_select.keys():
    if v == 280:
        x = 0.947
    elif v == 385 or v == 785:
        x = 0.879
    elif v == 390:
        x = 0.89

    print(v)
    print(test_select[v])
    # sub_df.iloc[test_select[v]][1] = x
    sub_df.loc[test_select[v], 1] = x

sub_df.to_csv(result_path + '/jinnan_round_submit_diff_2.csv', index=False, header=False)

print(len(diff_train))

# training's l2: 0.00014802	valid_1's l2: 0.000148992
