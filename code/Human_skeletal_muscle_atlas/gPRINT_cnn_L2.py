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



path='/data/yrj/MTJ/人类骨骼肌图谱/'
path_need=sys.argv[1]
path_d= path+path_need+'/'+sys.argv[2]+'_all.csv'  ##reference

out_way=path+path_need+'/L2/'
mkdir(out_way)

path_out= out_way+'out_'

#data= pd.read_csv("/data/yrj/download_sc/Pancreas/seurat/baron_all.csv",low_memory=False)##_new
data= pd.read_csv(path_d,low_memory=False)##_new
gene_name=data['Unnamed: 0' ]
data=data.drop(['Unnamed: 0'],axis=1)
data=data.T
data=pd.DataFrame(data)
data=data.sample(frac=1)
length=data.shape[1]-1

all_type=pd.value_counts(data.values[:,length])
#delet_type=all_type[all_type<10]
data_last_1=data.sample(frac=1)
data2_1=data_last_1.drop(data_last_1.columns[len(data_last_1.columns)-1], axis=1)

X_1 = np.expand_dims(data2_1.values.astype(float), axis=2)
Y_1 = data_last_1.values[:, length]
Y_1_name = np.unique(Y_1)
nclass = len(Y_1_name)

encoder = LabelEncoder()
Y_1_encoded = encoder.fit_transform(Y_1)
Y_1_onehot = np_utils.to_categorical(Y_1_encoded)
X_train=X_1
Y_train=Y_1_onehot


#####测试集

path_d_2= path+path_need+'/'+sys.argv[3]+'_all.csv'  ##reference
data_2= pd.read_csv(path_d_2,low_memory=False)
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

X_2 = np.expand_dims(data2_2.values.astype(float), axis=2)
Y_2 = data_last_2.values[:, length]
Y_2_name = np.unique(Y_2)
#nclass = len(Y_2_name)
#from sklearn.utils import shuffle
#X,Y = shuffle(X,Y, random_state=1337) 
nclass2 = len(Y_2_name)
# 湿度分类编码为数字
encoder = LabelEncoder()
Y_2_encoded = encoder.fit_transform(Y_2)
Y_2_onehot = np_utils.to_categorical(Y_2_encoded)

X_test=X_2
Y_test=Y_2_onehot






## 两层CNN
out_way=path+path_need+'/L2/'
mkdir(out_way)
path_out= out_way+'out_'
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
history=estimator.fit(X_train, Y_train)



########### 保存 #######

path_name=path_out+'model.json'
model_json = estimator.model.to_json()
with open(path_name,'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构


path_name=path_out+'model.h5'
estimator.model.save_weights(path_name)



########### 读取 #######
# 加载模型用做预测
path_name=path_out+'model.json'
json_file = open(path_name, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
path_name=path_out+'model.h5'
loaded_model.load_weights(path_name)



########predicted#######
predicted=estimator.model.predict(X_test)
predicted_label=estimator.model.predict_classes(X_test)

truelabel = Y_test.argmax(axis=-1) # 将one-hot转化为label

p = precision_score(truelabel, predicted_label, average='weighted')
r = recall_score(truelabel, predicted_label, average='weighted')
f1score = f1_score(truelabel, predicted_label, average='weighted')
print('precision_score: {:.2%}'.format(p))
print("recall_score: {:.2%}".format(r))
print("f1_score: {:.2%}".format(f1score))
###### ENCODE的顺序是按照np.unique 来的
train_correct_all =(Y_1_name[predicted_label]==Y_2_name[truelabel]).sum()
train_acc_all=train_correct_all/Y_test.shape[0]

print("all types' accuracy: {:.2%}".format(train_acc_all))

result_scores = [['precision',p],['recall',r],['f1_score',f1score],['accuracy',train_acc_all]]
result_scores=pd.DataFrame(result_scores)
path_name=path_out+sys.argv[2]+'result_all.csv'
result_scores.to_csv(path_name)
####
nclass2 = len(Y_2_name)
train_correct=dict()
train_acc=dict()
train_precision_score=dict()
train_recall_score=dict()
train_f1_score=dict()
train_accuracy_score=dict()
all_train=dict()
#true_sub[i]=truelabel[truelabel==i]

for i in range(nclass2):
	truelabel2=pd.DataFrame(truelabel)
	true3=truelabel2[truelabel2==i]
	t1=Y_1_name[predicted_label[true3.dropna().index]]
	t2=Y_2_name[truelabel[true3.dropna().index]]
	train_correct[i] =(t1==t2).sum()
	all_train[i]=len(t2)
	train_acc[i]=(t1==t2).sum()/len(t2)
	train_accuracy_score[i]=metrics.accuracy_score(t2, t1)
	train_precision_score[i]=metrics.precision_score(t2, t1, average='weighted')
	train_recall_score[i]=metrics.recall_score(t2, t1, average='weighted')
	train_f1_score[i]=metrics.f1_score(t2, t1, average='weighted')


accuracy=pd.DataFrame.from_dict(data=train_accuracy_score, orient='index')
precision=pd.DataFrame.from_dict(data=train_precision_score, orient='index')
recall=pd.DataFrame.from_dict(data=train_recall_score, orient='index')
f1=pd.DataFrame.from_dict(data=train_f1_score, orient='index')
accuracy.columns = ['accuracy']
accuracy=accuracy.sort_index(ascending=True)
precision.columns = ['precision']
precision=precision.sort_index(ascending=True)
recall.columns = ['recall']
recall=recall.sort_index(ascending=True)
f1.columns = ['f1']
f1=f1.sort_index(ascending=True)
result = pd.merge(precision,recall,left_index=True,right_index=True)
result = pd.merge(accuracy,precision,left_index=True,right_index=True)
result = pd.merge(result,recall,left_index=True,right_index=True)
result = pd.merge(result,f1,left_index=True,right_index=True)

result.index=Y_2_name

path_name=path_out+'result_log_file2.csv'
result.to_csv(path_name)


r1=Y_1_name[predicted_label]

r2=Y_2_name[truelabel]
meta=np.vstack((r1,r2))
meta=pd.DataFrame(meta)
meta.columns=data_last_2.index
meta.index=['predict','real']
path_name=path_out+'metadata2.csv'
meta.to_csv(path_name,sep=',',index=True,header=True)



