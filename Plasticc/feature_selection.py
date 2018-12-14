import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif


# Extracting features from existing dataset

train_df = pd.read_csv('training_set.csv')
id_list = train_df['object_id'].unique()
# print(len(id_list))
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
#full_train_ss = np.genfromtxt('full_train_ss.csv', delimiter=',')

# Use the extra tree classifier to rate feature importance
#model = ExtraTreesClassifier()
#model.fit(full_train_ss[:100], y[:100])

#print("EXTRA TREES FEATURE IMPORTANCE")
#for i in range(len(full_train.columns.values)):
#    print("{}: {}".format(full_train.columns.values[i], model.feature_importances_[i]))

# Get a smaller sample of the data
#chosen_indices = list(range(len(full_train_ss)))
#random.shuffle(chosen_indices)
#chosen_indices = chosen_indices[0:int(len(chosen_indices)/2)]

#full_train_ss_sample = [full_train_ss[index] for index in chosen_indices]
#y_sample = [y[index] for index in chosen_indices]

# Use f test and mutual information to look at the features importance
f_test, _ = f_classif(full_train_ss, y)
mi = mutual_info_classif(full_train_ss, y)

print(len(y))
print("\nF TESTS AND MUTUAL INFO")
for i in range(len(full_train.columns.values)):
    print("{}: f_test={}, mutual_info={}".format(full_train.columns.values[i], f_test[i], mi[i]))
















