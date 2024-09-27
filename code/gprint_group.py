import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


import math
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc
import itertools
from sklearn import metrics
import matplotlib as mpl
from itertools import cycle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from keras.layers import Input,Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

#### nohup python cnn_L2.py cnn_12vs17 wk12_14 wk17_18 &

### 创建文件夹的方法

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---") 
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---") 


os.chdir("..")  # change folder
print(os.getcwd())  

path = './match_model/'
path_main = path + sys.argv[1] + '_anno_' + sys.argv[2] + '/'

path_ref= path_main+'reference/'+sys.argv[1]+'_group_all.csv'  ##reference

out_way=path_main+'result/group_level/'


mkdir(out_way)

path_out= out_way+'out_'

#####Train Set

data= pd.read_csv(path_ref,low_memory=False)##_new
data=data.drop(['Unnamed: 0'],axis=1)
data=data.T
data=pd.DataFrame(data)
data=data.sample(frac=1)
length=data.shape[1]-1

all_type=pd.value_counts(data.values[:,length])
data_last_1=data.sample(frac=1)
data2_1=data_last_1.drop(data_last_1.columns[len(data_last_1.columns)-1], axis=1)


Y_1 = data_last_1.values[:, length]
Y_1_name = np.unique(Y_1)
nclass = len(Y_1_name)

encoder = LabelEncoder()
Y_1_encoded = encoder.fit_transform(Y_1)

data_train=data2_1.values.astype(float)
x_train = data_train.reshape(len(data_train),-1)
y_train = np_utils.to_categorical(Y_1_encoded)

#####Test Set

path_query= path_main+'query/'+sys.argv[2]+'_group_all.csv'  ##query
data_2= pd.read_csv(path_query,low_memory=False)
data_2=data_2.drop(['Unnamed: 0'],axis=1)
data_2=data_2.T
data_2=pd.DataFrame(data_2)
data_2=data_2.sample(frac=1)
#data_2.shape=(344, 32539)

length=data_2.shape[1]-1

all_type2=pd.value_counts(data_2.values[:,length])

data_last_2=data_2.sample(frac=1)
data2_2=data_last_2.drop(data_last_2.columns[len(data_last_2.columns)-1], axis=1)
pd.value_counts(data_last_2.values[:,length])
#data2_2_log=np.log10(data2_2.astype(float)+1)###log10(counts+1)

#X_2 = np.expand_dims(data2_2.values.astype(float), axis=2)
Y_2 = data_last_2.values[:, length]
Y_2_name = np.unique(Y_2)
#nclass = len(Y_2_name)
#from sklearn.utils import shuffle
#X,Y = shuffle(X,Y, random_state=1337) 
nclass2 = len(Y_2_name)
#
encoder = LabelEncoder()
Y_2_encoded = encoder.fit_transform(Y_2)
#Y_2_onehot = np_utils.to_categorical(Y_2_encoded)
data2=data2_2.values.astype(float)
x_test = data2.reshape(len(data2),-1)
y_test = np_utils.to_categorical(Y_2_encoded)


## BP

##################  model  ##############

# This returns a tensor
inputs = Input(shape=(length,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(nclass, activation='softmax')(x)

# This creates a model that includes the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

path_name=path_out+'model_classifier_bp.png'
plot_model(model, to_file=path_name, show_shapes=True)

model.fit(x_train, y_train,batch_size = 32, 
          epochs = 50, verbose=0)  # starts training



############# save  ###########
# 

path_name=path_out+'model_group.json'
model_json = model.to_json()
with open(path_name,'w')as json_file:
  json_file.write(model_json)


path_name=path_out+'model_group.h5'
model.save_weights(path_name)


########### load model #######

path_name=path_out+'model_group.json'
json_file = open(path_name, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
path_name=path_out+'model_group.h5'
loaded_model.load_weights(path_name)


########predicted#######

predicted=loaded_model.predict(x_test)
predicted_label=np.argmax(predicted,axis=1)


truelabel = y_test.argmax(axis=-1) # 

r1=Y_1_name[predicted_label]

r2=Y_2_name[truelabel]
meta=np.vstack((r1,r2))
meta=pd.DataFrame(meta)
meta.columns=data_last_2.index
meta.index=['predict_group','real']
path_name=out_way+'anno_result.csv'
meta.to_csv(path_name,sep=',',index=True,header=True)
print('The annotate result has been saved in:',path_name)
