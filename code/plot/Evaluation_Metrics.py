import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import math
import keras

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics

import os
import sys



path='/data/yrj/MTJ/人类骨骼肌图谱/cnn_yr7vs34/all_method/'

method='scmap_cell'
path_d=path+'all_anno.csv'
path_out= path+'result/'+method+'_result.csv'
#data= pd.read_csv("/data/yrj/download_sc/Pancreas/seurat/baron_all.csv",low_memory=False)##_new
data= pd.read_csv(path_d,low_memory=False)##_new
data=pd.DataFrame(data)
truelabel=data.iloc[:,12]
predicted_label=data.iloc[:,21]

accuracy = accuracy_score(truelabel, predicted_label)
p = precision_score(truelabel, predicted_label, average='weighted')
r = recall_score(truelabel, predicted_label, average='weighted')
f1score = f1_score(truelabel, predicted_label, average='weighted')
print('precision_score: {:.2%}'.format(p))
print("recall_score: {:.2%}".format(r))
print("f1_score: {:.2%}".format(f1score))

result_scores = [['precision',p],['recall',r],['f1_score',f1score],['accuracy',accuracy]]
result_scores=pd.DataFrame(result_scores)
path_name=path_out
result_scores.to_csv(path_name)
