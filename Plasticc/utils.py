import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
from sklearn.preprocessing import StandardScaler
import csv

import featuretools as ft

# Extracting features from existing dataset

train = pd.read_csv('training_set.csv')
id_list = train['object_id'].unique()
print(len(id_list))
pass_list = train['passband'].unique()
# aggregation_train = train_df.groupby('object_id').groupby('passband')

# Using Feature tools to create new entity set
entity_set = ft.EntitySet(id = 'Plasticc')

# Merging Data
meta = pd.read_csv('training_set_metadata.csv')
full_train = train.reset_index().merge(
    right = meta.reset_index(), 
    how = 'outer', 
    on = 'object_id'
)
if 'index_x' in full_train:
    del full_train['index_x']

if 'index_y' in full_train:
    del full_train['index_y']
# print(full_train.head())
# print(full_train.dtypes.sample(10))

if 'target' in full_train:
    y= full_train['target']
    del full_train['target']

classes = sorted(y.unique())

class_weight = {
    c: 1 for c in classes
}

for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes: ', classes)


train_mean = full_train.mean(axis = 0)
full_train.fillna(train_mean, inplace= True)

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state=1) 
print(full_train.info())
# Additional feature engineering
entity_set = entity_set.entity_from_dataframe(entity_id = 'Plasticc', dataframe= full_train,
                                              index = 'index', time_index = 'mjd')
print(entity_set['Plasticc'])
# Using feature tools for deep feature synthesis

feature_matrix, feature_names = ft.dfs(entityset = entity_set, target_entity = 'Plasticc', verbose=True,                                        agg_primitives = ['std'],
                                        trans_primitives=['percentile'] )
print(entity_set['Plasticc'])
print(feature_matrix.info())
print(feature_matrix.size)
ss = StandardScaler()
feature_matrix_ss = ss.fit_transform(feature_matrix)
print(feature_matrix_ss.size)

feature_matrix.to_csv('feature_matrix.csv')



















