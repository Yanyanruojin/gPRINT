print('OK_OK')
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

import os

os.chdir("..")  # change folder
print(os.getcwd())  

path_ref = './reference_model/Pancreas/model/cell_level/Pancreas_reference_celltype.csv'
Y_1 = pd.read_csv(path_ref,low_memory=False,header=None)
print('Y1 is OK')


