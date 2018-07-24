import lightgbm as lgb
import pandas as pd
import os
if lgb.compat.MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
else:
    raise ImportError('You need to install matplotlib for plot_example.py.')

file_path = os.path.abspath('..')
gbm = lgb.Booster(model_file=file_path + '/tmp/model.txt')

# print('Plot 84th tree...')  # one tree use categorical feature to split
# ax = lgb.plot_tree(gbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
# plt.show()

for i in range(100):
    print('Plot %dth tree with graphviz...' % i)
    graph = lgb.create_tree_digraph(gbm, tree_index=i, name='Tree' + str(i))
    graph.render(view=True)
