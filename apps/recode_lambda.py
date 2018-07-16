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
        if line[0] == '#':
            continue
        terms = line.split("\s+")
        if len(terms) == 3:
            params[terms[0]] = terms[2]
            print("key %s , value %s" % (terms[0], terms[2]))


# # load or create your dataset
# print('Load data...')
# df_train = pd.read_csv('regression.train', header=None, sep='\t')
# df_test = pd.read_csv('regression.test', header=None, sep='\t')

# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values

# # create dataset for lightgbm
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# # specify your configurations as a dict
# params = {
#     'num_leaves': 5,
#     'metric': ('l1', 'l2'),
#     'verbose': 0
# }

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