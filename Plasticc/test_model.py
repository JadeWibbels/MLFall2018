import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import featuretools as ft

#from starter import multi_weighted_logloss 

# Extracting features from existing dataset

test_df = pd.read_csv('test_set_sample.csv')
id_list = test_df['object_id'].unique()
print(len(id_list))
pass_list = test_df['passband'].unique()
# aggregation_train = train_df.groupby('object_id').groupby('passband')

# Using Feature tools to create new entity set
test_set = ft.EntitySet(id = 'Plasticc_Test')

# print(test_df.head())

# Merging Data
meta_test = pd.read_csv('test_set_metadata.csv')
full_test = test_df.reset_index().merge(
    right = meta_test, 
    how = 'inner', 
    on = 'object_id'
)

print(full_test.info())
# print(full_test.dtypes.sample(10))

if 'target' in full_test:
    y_test= full_test['target']
    del full_test['target']

#classes = sorted(y_test.unique())

#class_weight = {
#    c: 1 for c in classes
#}

#for c in [64, 15]:
#    class_weight[c] = 2

#print('Unique classes: ', classes)


test_mean = full_test.mean(axis = 0)
full_test.fillna(test_mean, inplace= True)

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state=1) 

# Additional feature engineering
test_set = test_set.entity_from_dataframe(entity_id = 'Plasticc_Test', dataframe= full_test,
                                              index = 'index', time_index = 'mjd')
#print(test_set['Plasticc_Test'])
# Using feature tools for deep feature synthesis

test_matrix, feature_names = ft.dfs(entityset = test_set, target_entity = 'Plasticc_Test',
                                 agg_primitives = ['mean', 'max', 'std', 'skew', 'percent_true'], 
                                 trans_primitives = ['and', 'percentile', 'time_since_previous'])
ss = StandardScaler()
test_matrix_ss = ss.fit_transform(test_matrix)
print(test_matrix.shape)
print(test_matrix_ss.shape)
#model = load_model('keras.model')
#oof_pres[val_, :] = model.predict_proba(classes, test_matrix_ss, verbose = 1)
#print(multi_weighted_logloss(model.predict_proba(test_matrix_ss, verbose = 1)))

#score = model.evaluate(test_matrix_ss, verbose=1)
#print(score[0])
#print( score[1])
#print(multi_weighted_logloss(model.predict_proba(test_matrix_ss, verbose = 1)))

