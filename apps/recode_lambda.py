# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
import os

file_path = os.path.abspath('.')
params = {}

print('loading conf...')
with open(file_path + '/tmp/train.conf', 'r') as f:
    for line in f.readlines():
        if line[0] == '#' or line[0] == '\n':
            continue
        terms = line.split(" ")
        params[terms[0]] = terms[2]
        print("key %s , value %s" % (terms[0], terms[2]))


# load or create your dataset
print('Load data...')
td_train = pd.read_csv('file_path/tmp/train_data', header=None, sep=',').values
td_query = pd.read_csv('file_path/tmp/train_data.query', header=None, sep=',').values
vd_train = pd.read_csv('file_path/tmp/vali_data', header=None, sep=',').values
vd_query = pd.read_csv('file_path/tmp/vali_data.query', header=None, sep=',').values


# create dataset for lightgbm
lgb_train = lgb.Dataset(td_train, group=td_query)
lgb_test = lgb.Dataset(vd_train, group=vd_query, reference=lgb_train)

# evals_result = {}  # to record eval results for plotting

# print('Start training...')
# # train
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=100,
#                 valid_sets=[lgb_train, lgb_test],
#                 feature_name=['f' + str(i + 1) for i in range(28)],
#                 categorical_feature=[21],
#                 evals_result=evals_result,
#                 verbose_eval=10)
