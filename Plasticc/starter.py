from keras.models import Sequential, load_model
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter
from sklearn.metrics import confusion_matrix

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Naive

#from utils2 import *
from utils import *

from test_model import *
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred):  
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss

def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


''' adapted from public kernel in starter kit
https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
'''
K.clear_session()
def create_model(dropout_rate=0.25,activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=feature_matrix_ss.shape[1], activation='relu'))
    #model.add(Dense(start_neurons, input_dim=full_train_ss.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate/2))
    
    model.add(Dense(len(classes), activation='softmax'))
    return model
unique_y = np.unique(y)
class_map = dict()
for i,val in enumerate(unique_y):
    class_map[val] = i
        
y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map)

y_count = Counter(y_map)
wtable = np.zeros((len(unique_y),))
for i in range(len(unique_y)):
    wtable[i] = y_count[i]/y_map.shape[0]


#log = ''
clfs = []
oof_preds = np.zeros((len(feature_matrix_ss), len(classes)))
#oof_preds = np.zeros((len(full_train_ss), len(classes)))
epochs = 200 
batch_size = 1000
for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    callback = EarlyStopping(monitor = 'val_loss', patience = 2)
    checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)

    x_train, y_train = feature_matrix_ss[trn_], y_categorical[trn_]
    x_valid, y_valid = feature_matrix_ss[val_], y_categorical[val_]
    #x_train, y_train = full_train_ss[trn_], y_categorical[trn_]
    #x_valid, y_valid = full_train_ss[val_], y_categorical[val_]
 

    model = create_model(dropout_rate=0.5,activation='tanh')    
    model.compile(loss=mywloss, optimizer='Adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,shuffle=True,verbose=0,callbacks=[checkPoint, callback])       
    
    
    print('Loading Best Model:')
    model.load_weights('./keras.model')
    # # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict_proba(x_valid,batch_size=batch_size)
    #log+= multi_weighted_logloss(y_valid, model.predict_proba(x_valid,batch_size=batch_size))+'\n'
    print(multi_weighted_logloss(y_valid, model.predict_proba(x_valid,batch_size=batch_size)))
    clfs.append(model)


print("saving Model")    
model.save('p2_model.h5')  
print("saving Model") 
np.append(unique_y,[99])   
predict = model.predict_proba(test_matrix_ss,batch_size=batch_size)
print(score[0])
print(score[1])

