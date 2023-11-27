# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import math
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc
import itertools
from sklearn import metrics
import matplotlib as mpl
from itertools import cycle

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics

import os
import sys

###  nohup python gPRINT_Internal_test.py input_path self out_path

##/data/yrj/download_sc/HCL/CNN/test_end_pos/cnn_self/
###########################################################

path = sys.argv[1]   ##路径
#path='/data/yrj/download_sc/tissue/Li/'
#data_path=path + 'li_data.csv'
data_path=path + sys.argv[2]+'_order_all.csv'
path_out=sys.argv[3]

data= pd.read_csv(data_path, encoding='gbk', low_memory=False)
data=data.drop(['Unnamed: 0'],axis=1)
data=data.T
data=pd.DataFrame(data)
data=data.sample(frac=1)
#data.shape=(344, 32539)
length=data.shape[1]-1

all_type=pd.value_counts(data.values[:,length])
delet_type=all_type[all_type<10]
delet_type.index
#Index(['unclear', 'epsilon'], dtype='object')
len(delet_type)
#2

#data_last=data.loc[pd.Index(data.values[:,length]).difference(delet_type.index)]
#X2=data.drop([length],axis=1)
#data2=data_last.drop([length],axis=1)

data_last=data

data2=data_last.drop(data_last.columns[len(data_last.columns)-1], axis=1)
X = np.expand_dims(data2.values.astype(float), axis=2)
Y = data_last.values[:, length]
Y_name = np.unique(Y)
nclass = len(Y_name)
#from sklearn.utils import shuffle
#X,Y = shuffle(X,Y, random_state=1337) 




from sklearn.utils import resample

sample_size=data_last.shape[0]//nclass
if X[Y==Y_name[0]].shape[0] > sample_size :
	X_balanced, Y_balanced = resample(X[Y==Y_name[0]], Y[Y==Y_name[0]], replace=True, n_samples=sample_size)

X_balanced=X[Y==Y_name[0]]
Y_balanced=Y[Y==Y_name[0]]
for i in range(1,nclass):
	if X[Y==Y_name[i]].shape[0]> sample_size :
		X_balanced = np.concatenate([X_balanced,resample(X[Y==Y_name[i]], replace=True, n_samples=sample_size)], axis=0)
		Y_balanced = np.concatenate([Y_balanced,resample(Y[Y==Y_name[i]], replace=True, n_samples=sample_size)], axis=0)
	
	else:
		X_balanced = np.concatenate([X_balanced,X[Y==Y_name[i]]], axis=0)
		Y_balanced = np.concatenate([Y_balanced,Y[Y==Y_name[i]]], axis=0)




# 分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y_balanced)
Y_onehot = np_utils.to_categorical(Y_encoded)

X_train, X_test, Y_train, Y_test = train_test_split(X_balanced, Y_onehot, test_size=0.3, random_state=0)


############## resample ###############

from sklearn.utils import resample

sample_size=data_last.shape[0]//nclass
if X[Y==Y_name[0]].shape[0] > sample_size :
	X_balanced, Y_balanced = resample(X[Y==Y_name[0]], Y[Y==Y_name[0]], replace=True, n_samples=sample_size)

X_balanced=X[Y==Y_name[0]]
Y_balanced=Y[Y==Y_name[0]]
for i in range(1,nclass):
	if X[Y==Y_name[i]].shape[0]> sample_size :
		X_balanced = np.concatenate([X_balanced,resample(X[Y==Y_name[i]], replace=True, n_samples=sample_size)], axis=0)
		Y_balanced = np.concatenate([Y_balanced,resample(Y[Y==Y_name[i]], replace=True, n_samples=sample_size)], axis=0)
	
	else:
		X_balanced = np.concatenate([X_balanced,X[Y==Y_name[i]]], axis=0)
		Y_balanced = np.concatenate([Y_balanced,Y[Y==Y_name[i]]], axis=0)




# 分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y_balanced)
Y_onehot = np_utils.to_categorical(Y_encoded)

X_train, X_test, Y_train, Y_test = train_test_split(X_balanced, Y_onehot, test_size=0.3, random_state=0)


#####################

# 定义神经网络
#nclass=14
def baseline_model():
	model = Sequential()
	model.add(Conv1D(16, 3, input_shape=(length, 1)))
	model.add(Conv1D(16, 3, activation='tanh'))
	model.add(MaxPooling1D(3))
	model.add(Conv1D(64, 3, activation='tanh'))
	model.add(Conv1D(64, 3, activation='tanh'))
	model.add(MaxPooling1D(3))
	model.add(Conv1D(64, 3, activation='tanh'))
	model.add(Conv1D(64, 3, activation='tanh'))
	model.add(MaxPooling1D(3))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(nclass, activation='softmax'))
	path_name=path_out+sys.argv[2]+'_model_classifier.png'
	plot_model(model, to_file=path_name, show_shapes=True)
	print(model.summary())
	model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
	##optimizer='adam'
	return model
 



estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=32, verbose=1)
history=estimator.fit(X_train, Y_train)

# 分类准确率
print("The accuracy of the classification model:")
scores = estimator.model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (estimator.model.metrics_names[1], scores[1] * 100))

##五折交叉验证
#cnn_scores = cross_val_score(estimator, X_train, Y_train, cv=5)
#print(cnn_scores)

#path_name=path_out+sys.argv[2]+'_5fold.csv'
#cnn_scores2=pd.DataFrame(cnn_scores)
#cnn_scores2.to_csv(path_name)

############# 保存 ###########
# 将其模型转换为json
#path_out= sys.argv[4]

path_name=path_out+sys.argv[2]+'_model.json'
model_json = estimator.model.to_json()
with open(path_name,'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构


path_name=path_out+sys.argv[2]+'_model.h5'
estimator.model.save_weights(path_name)


########### 读取 #######
# 加载模型用做预测
path_name=path_out+sys.argv[2]+'_model.json'
json_file = open(path_name, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
path_name=path_out+sys.argv[2]+'_model.h5'
loaded_model.load_weights(path_name)


########predicted#######
predicted=estimator.model.predict(X_test)
predicted_label=estimator.model.predict_classes(X_test)

truelabel = Y_test.argmax(axis=-1) # 将one-hot转化为label

p = precision_score(truelabel, predicted_label, average='weighted')
r = recall_score(truelabel, predicted_label, average='weighted')
f1score = f1_score(truelabel, predicted_label, average='weighted')
print(path_out)
print('precision_score: {:.2%}'.format(p))
print("recall_score: {:.2%}".format(r))
print("f1_score: {:.2%}".format(f1score))

###### ENCODE的顺序是按照np.unique 来的
train_correct_all =(Y_name[predicted_label]==Y_name[truelabel]).sum()
train_acc_all=train_correct_all/Y_test.shape[0]

print("all types' accuracy: {:.2%}".format(train_acc_all))

result_scores = [['precision',p],['recall',r],['f1_score',f1score],['accuracy',train_acc_all]]
result_scores=pd.DataFrame(result_scores)
path_name=path_out+sys.argv[2]+'_result_all.csv'
result_scores.to_csv(path_name)

#cnn_scores2.to_csv(path_name)

####想要 每个细胞类别对应的正确率（Y_2 的type A [可能对应的是1]，对应到Y_1的type A[可能对应的是2]）
#for i in range(nclass2):
	#train_correct[i] =(Y_1_name[i]==Y_2_name[i]).sum()/Y_test.shape[0]

####
nclass= len(Y_name)
train_correct=dict()
train_acc=dict()
train_precision_score=dict()
train_recall_score=dict()
train_f1_score=dict()
train_accuracy_score=dict()
all_train=dict()
#true_sub[i]=truelabel[truelabel==i]

for i in range(nclass):
	truelabel2=pd.DataFrame(truelabel)
	true3=truelabel2[truelabel2==i]
	#true3[i].dropna().index
	t1=Y_name[predicted_label[true3.dropna().index]]
	t2=Y_name[truelabel[true3.dropna().index]]
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
#result = pd.merge(precision,recall,left_index=True,right_index=True)
result = pd.merge(accuracy,precision,left_index=True,right_index=True)
result = pd.merge(result,recall,left_index=True,right_index=True)
result = pd.merge(result,f1,left_index=True,right_index=True)

result.index=Y_name

path_name=path_out+sys.argv[2]+'_result.csv'
result.to_csv(path_name)




# 卷积网络可视化
def visual(model, data, num_layer=1):
	# data:图像array数据
	# layer:第n层的输出
	data = np.expand_dims(data, axis=0)     # 开头加一维
	layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
	f1 = layer([data])[0]
	num = f1.shape[-1]
	plt.figure(figsize=(8, 8))
	for i in range(num):
		plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
		plt.imshow(f1[0, :, :, i] * 255, cmap='gray')
		plt.axis('off')
	
	path_name=path+'_layer_'+num_layer+'_CNN.png'
	plt.savefig(path_name, dpi=200, bbox_inches='tight', transparent=False)
	plt.show()


#visual(loaded_model, data2, 1)	# 卷积层


# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Adipose_Confusion matrix',cmap=plt.cm.jet):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号	
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks,np.unique(Y))
	plt.yticks(tick_marks,np.unique(Y))
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('real')
	plt.xlabel('predict')
	path_name=path_out+sys.argv[2]+'_confusion_matrix.png'
	plt.savefig(path_name, dpi=200, bbox_inches='tight', transparent=False)


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
	predictions = model.predict_classes(x_val)
	truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
	conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
	plt.figure()
	plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))


#显示混淆矩阵
plot_confuse(estimator.model, X_test, Y_test)

Y_score=estimator.model.predict(X_test)


###AUC,ROC的绘制

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(nclass):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nclass)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nclass):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


# Finally average it and compute AUC
mean_tpr /= nclass
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nclass), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
path_name=path_out+sys.argv[2]+'_ROC.png'

plt.savefig(path_name, dpi=200, bbox_inches='tight', transparent=False)






