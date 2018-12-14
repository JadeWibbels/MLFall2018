import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
from sklearn.preprocessing import StandardScaler

# Extracting features from existing dataset

train_df = pd.read_csv('training_set.csv')
id_list = train_df['object_id'].unique()
pass_list = train_df['passband'].unique()
# aggregation_train = train_df.groupby('object_id').groupby('passband')

# Merging Data
meta_train = pd.read_csv('training_set_metadata.csv')
full_train = train_df.reset_index().merge(
    right = meta_train, 
    how = 'outer', 
    on = 'object_id'
)


if 'target' in full_train:
    y= full_train['target']
    del full_train['target']

classes = sorted(y.unique())

class_weight = {
    c: 1 for c in classes
}

for c in [64, 15]:
    class_weight[c] = 2



train_mean = full_train.mean(axis = 0)
full_train.fillna(train_mean, inplace= True)

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state=1) 

ss = StandardScaler()
full_train_ss = ss.fit_transform(full_train)

np.savetxt("full_train_ss.csv", full_train_ss, delimiter=",")



















