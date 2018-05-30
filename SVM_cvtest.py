import pandas as pd
import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import csv
import os
# import some data to play with
path = '/Users/Mickey/Desktop/Group/Term_Data/'

all_Feat = pd.read_csv(path+'T1_T2_Vol_termandpreterm.csv')
#all_Feat = all_Feat[all_Feat.columns[1:]]

labels = pd.read_csv(path+'BirthGA_termandpreterm.csv')
#labels = labels[labels.columns[2]]


x_all = all_Feat
#x_all = all_Feat
y_all = labels

X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.5,random_state = 42)

svm = SVR(kernel= 'rbf', C=1000,gamma = 0.001)
svm.fit(X,y)
pre = svm.predict(X_test)

curr_score = svm.score(X_test, y_test)
curr_cross_score = cross_val_score(svm, x_all, y_all,cv = 5)
curr_mean = np.mean(curr_cross_score)
curr_std = np.std(curr_cross_score)

print curr_score
print curr_cross_score
print curr_mean
print curr_std
