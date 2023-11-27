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

################        python          ##################


###  nohup python External_test.py input_path reference test out_path


###########################################################

path = sys.argv[1]   ##路径
path_d= path+ sys.argv[2] +'_all.csv' ##reference
#data= pd.read_csv("/data/yrj/download_sc/Pancreas/seurat/baron_all.csv",low_memory=False)##_new
data= pd.read_csv(path_d,low_memory=False)##_new
data=data.drop(['Unnamed: 0'],axis=1)
data=data.T
data=pd.DataFrame(data)
data=data.sample(frac=1)
length=data.shape[1]-1
#data.loc[data[(data[length]=='activated_stellate')|(data[length]=='quiescent_stellate')].index,length]= 'PSC'
#data.loc[data[data[dd]=='lymphoma cell line'].index,dd]='lymphom cell line'

#data.shape=(344, 32539)
#length=data.shape[1]-1

all_type=pd.value_counts(data.values[:,length])
#delet_type=all_type[all_type<10]
#delet_type.index

#Index(['unclear', 'epsilon'], dtype='object')
#len(delet_type)
#2
#
# &( data.values[:,length]!='epsilon')]
#data_last=data[(data.values[:,length]!=delet_type.index[0]) &( data.values[:,length]!=delet_type.index[1])]
#test_type=(data.values[:,length]!=delet_type.index[0])
#for i in range(len(delet_type)):
  #test_type=test_type&( data.values[:,length]!=delet_type.index[i])


#data_last_1=data[test_type]

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


################test############

#按照参考数据集合进行筛选基因

data_path_2=path +sys.argv[3]+'_all.csv'
data_2= pd.read_csv(data_path_2,low_memory=False)
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


##############
path_out= sys.argv[4]


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
	model.add(Flatten())
	model.add(Dense(nclass, activation='softmax'))
	path_name=path_out+'model_classifier2.png'
	plot_model(model, to_file=path_name, show_shapes=True)
	print(model.summary())
	model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
	##optimizer='adam'
	return model
 


estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=16, verbose=1)
history=estimator.fit(X_train, Y_train)

############# 保存 ###########
# 将其模型转换为json
#path_out= sys.argv[4]

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
predicted=loaded_model.predict(X_test)
predicted_label=loaded_model.predict_classes(X_test)

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


####想要 每个细胞类别对应的正确率（Y_2 的type A [可能对应的是1]，对应到Y_1的type A[可能对应的是2]）
#for i in range(nclass2):
	#train_correct[i] =(Y_1_name[i]==Y_2_name[i]).sum()/Y_test.shape[0]

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
	#true3[i].dropna().index
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

path_name=path_out+'result_log_file.csv'
result.to_csv(path_name)


####predicted_label 的 index和truelabel 的index一致
####
fpr = dict()
tpr = dict()
roc_auc = dict()
Y_score=predicted
for i in range(nclass2):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area（方法二）
#fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nclass2)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nclass2):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nclass2
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nclass2), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
path_name=path_out+'ROC.png'
plt.savefig(path_name, dpi=200, bbox_inches='tight', transparent=False)
plt.show()





# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
	plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号	
	plt.colorbar()
	tick_marks= np.arange(len(classes))
	tick_marks_2 = np.arange(nclass+1)
	plt.xticks(tick_marks_2,Y_1_name)
	plt.yticks(tick_marks,Y_2_name)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.xlabel('predict')
	plt.xticks(rotation=70)
	plt.ylabel('real')
	path_name=path_out+'confusion_matrix.png'
	plt.savefig(path_name, dpi=200, bbox_inches='tight', transparent=False)
	plt.show()
 
# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))
 


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
	predictions = model.predict_classes(x_val)
	truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
	conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
	plt.figure()
	plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))



plot_confuse(loaded_model, X_test, Y_test)


r1=Y_1_name[predicted_label]

r2=Y_2_name[truelabel]
meta=np.vstack((r1,r2))
meta=pd.DataFrame(meta)
meta.columns=data_last_2.index
meta2.index=['predict_bp','real']
path_name=path_out+'metadata.csv'
meta2.to_csv(path_name,sep=',',index=True,header=True)

