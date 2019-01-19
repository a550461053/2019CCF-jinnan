# coding=utf-8

# @Time    : 2019-01-11 11:13
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

# data path
pre_root_path = "data/pre-data"
result_path = "result"
train = pd.read_csv(pre_root_path + '/jinnan_round1_train_20181227.csv', encoding = 'gb18030')
test  = pd.read_csv(pre_root_path + '/jinnan_round1_testA_20181227.csv', encoding = 'gb18030')
print('load data')

stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]

print(stats_df)
print('##############')
target_col = "收率"

pre_show = False
if pre_show:
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train.shape[0]), np.sort(train[target_col].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('yield', fontsize=12)
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.distplot(train[target_col].values, bins=50, kde=False, color="red")
    plt.title("Histogram of yield")
    plt.xlabel('yield', fontsize=12)
    plt.show()

######################### 特征工程


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



time_second_columns = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for f in time_second_columns:
    try:
        data[f] = data[f].apply(timeTranSecond)
    except:
        print(f, '应该在前面被删除了！')

def  trans_temper(x):
    if not x:
        return 0
    return x + 273.15
    # return x * 9 / 5.0 + 32.0

# 温度转换为绝对零度
temp_columns = ['A6', 'A8', 'A10', 'A12', 'A15', 'A17', 'A21', 'A25', 'A27', 'B6', 'B8']
# for f in temp_columns:
#     try:
#         data[f] = data[f].apply(trans_temper)
#     except:
#         print(f, '出错了')


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

time_columns = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
for f in time_columns:
    data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)


data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

categorical_columns = [f for f in data.columns if f not in ['样本id']]
# categorical_columns = ['样本id']
# categorical_columns.extend(time_second_columns)
# categorical_columns.extend(time_columns)
# categorical_columns.extend(temp_columns)
numerical_columns = [f for f in data.columns if f not in categorical_columns]
print('categorical:', categorical_columns)
print('numerical:', numerical_columns)
# input()

# 有风的冬老哥，在群里无意爆出来的特征，让我提升了三个个点，当然也可以顺此继续扩展
data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
data['A3/a1_a3_a4_a19_b1_b12'] = data['A3']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
data['A4/a1_a3_a4_a19_b1_b12'] = data['A4']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
data['A19/a1_a3_a4_a19_b1_b12'] = data['A19']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
# 全不要：lgb:0.00012559, xgb:0.00012087, stack:0.00011704
numerical_columns.append('b14/a1_a3_a4_a19_b1_b12') # 0.0011582
# numerical_columns.append('A3/a1_a3_a4_a19_b1_b12') # 0。00011619
# numerical_columns.append('A4/a1_a3_a4_a19_b1_b12') # 注释之后：lgb:0.00012259, xgb:0.00012024, stack:0.00011627
# numerical_columns.append('A19/a1_a3_a4_a19_b1_b12') #

del data['A1']
del data['A2']  # A2+A3+A4是定值
del data['A3']
del data['A4']
categorical_columns.remove('A1')
categorical_columns.remove('A2')  # A2+A3+A4是定值
categorical_columns.remove('A3')
categorical_columns.remove('A4')

# numerical_columns.remove('A1')
# numerical_columns.remove('A2')
# numerical_columns.remove('A3')
# numerical_columns.remove('A4')

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
################################ 特征工程 end

# print('save 特征工程后的数据')
# print(X_train)
# # print(X_train[0])
# print(X_train[0][0])
# print(X_train[(0, 0)])
# X_train.to_csv(pre_root_path + '/jinnan_train_after_feature_selected.csv')
# X_test.to_csv(pre_root_path + '/jinnan_test_after_feature_selected.csv')
# print('save done!')
# input('input any key to start train.')

# 训练模型
#### lgb
print('lgb...')
param = {'num_leaves': 120,
         'min_data_in_leaf': 30,
         'objective': 'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 30,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=200)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
res_lgb = mean_squared_error(oof_lgb, target)
print("CV score: {:<8.8f}".format(res_lgb))

##### xgb
print('xgboost...')
xgb_params = {'eta': 0.005,
              'max_depth': 10,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'silent': True,
              'nthread': 4
              }

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=200, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
res_xgb = mean_squared_error(oof_xgb, target)
print("CV score: {:<8.8f}".format(res_xgb))


# 模型融合
# 将lgb和xgb的结果进行stacking
print('stacking...')
train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

res_stack = mean_squared_error(target.values, oof_stack)

print('lgb:{:<8.8f}, xgb:{:<8.8f}, stack:{:<8.8f}'.format(res_lgb, res_xgb, res_stack))

# 保存
sub_df = pd.read_csv(pre_root_path + '/jinnan_round1_submit_20181227.csv', header=None)
sub_df[1] = predictions
# sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))  # 这是覆盖读取的文件
sub_df.to_csv(result_path + '/jinnan_round1_submit_20181227_1.csv', index=0, header=0)  # 这是另存为，不保存索引行
print('save done!')
# sub_df.to_csv(result_path + '/jinnan_round1_submit_20181227_2.csv', index=0)  # 这是另存为，不保存索引行
#
# sub_df.to_csv(result_path + '/jinnan_round1_submit_20181227_3.csv')  # 这是另存为，不保存索引行
# train-rmse:0.005104	valid_data-rmse:0.010186
# train-rmse:0.005362	valid_data-rmse:0.010073

# 强特征：样本id提取数值

# 2019-01-13
print('采用后向搜索法进行特征选择')
def modeling_cross_validation(params, X, y, nr_folds=5):
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    folds = KFold(n_splits=nr_folds, shuffle=False, random_state=4096)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
        val_data = lgb.Dataset(X[val_idx], y[val_idx])

        num_round = 20000
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds=100)
        oof_preds[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)

    score = mean_squared_error(oof_preds, target)

    return score / 2

def featureSelect(init_cols):
    params = {'num_leaves': 120,
              'min_data_in_leaf': 30,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': 0.05,
              "min_child_samples": 30,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              "metric": 'mse',
              "lambda_l1": 0.02,
              "verbosity": -1}
    best_cols = init_cols.copy()
    best_score = modeling_cross_validation(params, train[init_cols].values, target.values, nr_folds=5)
    print("初始CV score: {:<8.8f}".format(best_score))
    for f in init_cols:

        best_cols.remove(f)
        score = modeling_cross_validation(params, train[best_cols].values, target.values, nr_folds=5)
        diff = best_score - score
        print('-' * 10)
        if diff > 0.0000002:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 有效果,删除！！".format(f, score, best_score))
            best_score = score
        else:
            print("当前移除特征: {}, CV score: {:<8.8f}, 最佳cv score: {:<8.8f}, 没效果,保留！！".format(f, score, best_score))
            best_cols.append(f)
    print('-' * 10)
    print("优化后CV score: {:<8.8f}".format(best_score))

    return best_cols

# best_features = featureSelect(train.columns.tolist())
# print(best_features)
# ['样本id', 'A5', 'A7', 'A9', 'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A19', 'A20', 'A21', 'A22', 'A24', 'A25', 'A26', 'A27', 'A28', 'B1', 'B4', 'B5', 'B8', 'B10', 'B11', 'B12', 'B14', 'b14/a1_a3_a4_a19_b1_b12', 'B14_to_A5_intTarget_0.0_mean', 'B14_to_A5_intTarget_1.0_mean', 'B14_to_A5_intTarget_2.0_mean', 'B14_to_A5_intTarget_4.0_mean', 'B14_to_A6_intTarget_0.0_mean', 'B14_to_A6_intTarget_1.0_mean', 'B14_to_A6_intTarget_2.0_mean', 'B14_to_A6_intTarget_3.0_mean', 'B14_to_A6_intTarget_4.0_mean', 'B14_to_A7_intTarget_0.0_mean', 'B14_to_A7_intTarget_1.0_mean', 'B14_to_A7_intTarget_2.0_mean', 'B14_to_A7_intTarget_3.0_mean', 'B14_to_A7_intTarget_4.0_mean', 'B14_to_A9_intTarget_0.0_mean', 'B14_to_A9_intTarget_1.0_mean', 'B14_to_A9_intTarget_2.0_mean', 'B14_to_A9_intTarget_3.0_mean', 'B14_to_A9_intTarget_4.0_mean', 'B14_to_A11_intTarget_0.0_mean', 'B14_to_A11_intTarget_1.0_mean', 'B14_to_A11_intTarget_2.0_mean', 'B14_to_A11_intTarget_3.0_mean', 'B14_to_A11_intTarget_4.0_mean', 'B14_to_A14_intTarget_0.0_mean', 'B14_to_A14_intTarget_1.0_mean', 'B14_to_A14_intTarget_2.0_mean', 'B14_to_A14_intTarget_3.0_mean', 'B14_to_A14_intTarget_4.0_mean', 'B14_to_A16_intTarget_0.0_mean', 'B14_to_A16_intTarget_1.0_mean', 'B14_to_A16_intTarget_2.0_mean', 'B14_to_A16_intTarget_3.0_mean', 'B14_to_A16_intTarget_4.0_mean', 'B14_to_A24_intTarget_0.0_mean', 'B14_to_A24_intTarget_1.0_mean', 'B14_to_A24_intTarget_2.0_mean', 'B14_to_A24_intTarget_3.0_mean', 'B14_to_A24_intTarget_4.0_mean', 'B14_to_A26_intTarget_0.0_mean', 'B14_to_A26_intTarget_1.0_mean', 'B14_to_A26_intTarget_2.0_mean', 'B14_to_A26_intTarget_3.0_mean', 'B14_to_A26_intTarget_4.0_mean', 'B14_to_B1_intTarget_0.0_mean', 'B14_to_B1_intTarget_1.0_mean', 'B14_to_B1_intTarget_2.0_mean', 'B14_to_B1_intTarget_3.0_mean', 'B14_to_B1_intTarget_4.0_mean', 'B14_to_B5_intTarget_0.0_mean', 'B14_to_B5_intTarget_1.0_mean', 'B14_to_B5_intTarget_2.0_mean', 'B14_to_B5_intTarget_3.0_mean', 'B14_to_B5_intTarget_4.0_mean', 'B14_to_B6_intTarget_0.0_mean', 'B14_to_B6_intTarget_1.0_mean', 'B14_to_B6_intTarget_2.0_mean', 'B14_to_B6_intTarget_3.0_mean', 'B14_to_B6_intTarget_4.0_mean', 'B14_to_B7_intTarget_0.0_mean', 'B14_to_B7_intTarget_1.0_mean', 'B14_to_B7_intTarget_2.0_mean', 'B14_to_B7_intTarget_3.0_mean', 'B14_to_B7_intTarget_4.0_mean', 'B14_to_B8_intTarget_0.0_mean', 'B14_to_B8_intTarget_1.0_mean', 'B14_to_B8_intTarget_2.0_mean', 'B14_to_B8_intTarget_3.0_mean', 'B14_to_B8_intTarget_4.0_mean', 'B14_to_B14_intTarget_0.0_mean', 'B14_to_B14_intTarget_1.0_mean', 'B14_to_B14_intTarget_2.0_mean', 'B14_to_B14_intTarget_3.0_mean', 'B14_to_B14_intTarget_4.0_mean']


# mark 结果
# 1. lgb: 0.00012278   xgb:0.00011954  stack:0.001158
# 2. lgb: 0.00012841  xgb：0.00011954  random改为了2019.。。。 stack: 0.000117123398856
# 3. 修改异常值范围是0.80，结果很差：lgb：0.00013963，xgb：0.00014100，stack：0.0013295
# 4. 修改异常值范围是0.85，结果是：lgb:0.00013454, xgb:0.00013631, stack:0.00012853
# 5. 修改异常值范围是0.87，结果是：lgb:0.00012841, xgb:0.00011954, stack:0.00011712
# 6. 改为0.87异常值，random改为2018：lgb:0.00012278, xgb:0.00011954, stack:0.00011582  ！！best
# 7. 加入A2特征，效果反而下降：lgb:0.00012284, xgb:0.00012004, stack:0.00011615
# 8. 去掉A2，去掉A6，效果下降：lgb:0.00012437, xgb:0.00012155, stack:0.00011769
# 9. 去掉A8或者A9，效果略微下降：lgb:0.00012263, xgb:0.00011982, stack:0.00011596
# lgb:0.00012278, xgb:0.00011954, stack:0.00011582