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


### Create a folder

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #Judge whether a folder exists and create it if it does not exist
		os.makedirs(path)            #makedirs creates a file if the path does not exist.
		print("---  new folder...  ---") 
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---") 


os.chdir("..")  # change folder
print(os.getcwd())  

path_first = './reference_model/'


path= path_first +sys.argv[2]+'/'


out_way=path+'result/cell_level/'+sys.argv[1]+'/'
mkdir(out_way)

path_out= path+'result/cell_level/'+sys.argv[1]+'/out_'


#### reference ####
path_ref = path_first +sys.argv[2]+'/model/cell_level/'+sys.argv[2]+'_reference_celltype.csv'
Y_1 = pd.read_csv(path_ref,low_memory=False,header=None)

Y_1_name = np.unique(Y_1)



#### query ####
path_d_2=path+'query/'+sys.argv[1]+'_query_all.csv'  ##reference

data_2= pd.read_csv(path_d_2,low_memory=False)
data_2=data_2.drop(['Unnamed: 0'],axis=1)
data_2=data_2.T
data_2=pd.DataFrame(data_2)
data_2=data_2.sample(frac=1)
length=data_2.shape[1]-1

all_type2=pd.value_counts(data_2.values[:,length])

data_last_2=data_2.sample(frac=1)
data2_2=data_last_2.drop(data_last_2.columns[len(data_last_2.columns)-1], axis=1)
pd.value_counts(data_last_2.values[:,length])
#data2_2_log=np.log10(data2_2.astype(float)+1)###log10(counts+1)

X_2 = np.expand_dims(data2_2.values.astype(float), axis=2)
Y_2 = data_last_2.values[:, length]
Y_2_name = np.unique(Y_2)


#nclass = len(Y_2_name)
#from sklearn.utils import shuffle
#X,Y = shuffle(X,Y, random_state=1337) 
nclass2 = len(Y_2_name)
encoder = LabelEncoder()
Y_2_encoded = encoder.fit_transform(Y_2)
Y_2_onehot = np_utils.to_categorical(Y_2_encoded)

X_test=X_2
Y_test=Y_2_onehot






## 两层CNN

def baseline_model():
	model = Sequential()
	model.add(Conv1D(16, 3, input_shape=(length, 1)))
	model.add(Conv1D(16, 3, activation='tanh'))
	model.add(MaxPooling1D(3))
	model.add(Flatten())
	model.add(Dense(nclass, activation='softmax'))
	path_name=path_out+'model_classifier.png'
	plot_model(model, to_file=path_name, show_shapes=True)
	print(model.summary())
	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
	##optimizer='adam'
	return model



estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=16, verbose=1)



########### 读取 #######
# 加载模型用做预测
path_model = path_first + sys.argv[2]+'/model/cell_level/'
path_name=path_model+'out_model.json'
json_file = open(path_name, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
path_name=path_model+'out_model.h5'
loaded_model.load_weights(path_name)


########predicted#######
predicted=loaded_model.predict(X_test)
predicted_label=loaded_model.predict_classes(X_test)

truelabel = Y_test.argmax(axis=-1) # 将one-hot转化为label


r1=Y_1_name[predicted_label]

r2=Y_2_name[truelabel]
meta=np.vstack((r1,r2))
meta=pd.DataFrame(meta)
meta.columns=data_last_2.index
meta.index=['predict','real']
path_name=out_way+'anno_result.csv'
meta.to_csv(path_name,sep=',',index=True,header=True)
print('The annotate result has been saved in:',path_name)

