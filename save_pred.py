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

best_kernel = 'rbf'
best_C = 100
best_gamma = 0.001

worst_kernel = 'linear'
worst_C = 100

path = '/Users/Mickey/Desktop/Group/All_Data/'

all_Feat = pd.read_csv(path+'T1_T2_Vol.csv')

print all_Feat.shape


all_Feat = all_Feat[all_Feat.columns[1:]]

labels = pd.read_csv(path+'ScanGA.csv')
print labels.shape
labels = labels[labels.columns[2]]

x_all = all_Feat
y_all = labels

X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.9,random_state = 42)
print '0.9'
print y_test
print np.shape(y_test)


pred = []
predb = []
labw = []
labb = []
### BEST
svm_best = SVR(kernel=best_kernel, C=best_C,gamma=best_gamma)
svm_best.fit(X,y)
pre_best = svm_best.predict(X_test)
lw = 2

print np.shape(pre_best)
print np.shape(y_test)
print type(pre_best)
print type(y_test)

pred.append(pd.DataFrame(pre_best))
pred = pd.concat(pred)
pred.to_csv('bestpred_SVM_SA_FullData.csv')

labb.append(pd.DataFrame(y_test))
labb = pd.concat(labb)
labb.to_csv('bestpredLABELS_SVM_SA_FullData.csv')

#pre_best.to_csv('bestpred_SVM_BA_FullData.csv')
#y_test.to_csv('LABELS_SVM_BA_FullData.csv')
#WORS
'''
svm_worst = SVR(kernel=worst_kernel, C=worst_C)
svm_worst.fit(X,y)
pre_worst = svm_worst.predict(X_test)
lw = 2
pre_worst.to_csv('worstpred_SVM_BA_FullData.csv',header=None)
'''
