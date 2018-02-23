# AdaBoost classification - use for feature extraction then svm
import pandas as pd
import nibabel
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile

from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# X, y = make_gaussian_quantiles(n_samples=1000, n_features=10,
#                                n_classes=3, random_state=1)
path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"

T1 = pd.read_csv(path+'T1.csv')
print 'Shape T1 : ', T1.shape
T2 = pd.read_csv(path+'T2.csv')
print 'Shape T2 : ', T2.shape
Volume = pd.read_csv(path+'Volume.csv')
print 'Shape vol : ', Volume.shape
PatCode = pd.read_csv(path+'PatCode.csv')
print 'Shape PatCode : ', PatCode.shape
SessID = pd.read_csv(path+'SessID.csv')
print 'Shape SessID : ', SessID.shape
BirthAge = pd.read_csv(path+'BirthAge.csv')
print 'Shape BA : ', BirthAge.shape
ScanAge = pd.read_csv(path+'ScanAge.csv')
print 'Shape SA : ', ScanAge.shape
Gender = pd.read_csv(path+'Gender.csv')
print 'Shape Gender : ', Gender.shape

T1 = T1[T1.columns[2:]]
Gender = Gender[Gender.columns[1]]
T2 = T2[T2.columns[2:]]
SA = ScanAge[ScanAge.columns[1]]

T = pd.concat([T1, T2], axis = 1)
print T.shape
# for row in T1.itertuples():
#     if i % 2 == 0:
#         train_t1.append(pd.DataFrame([row]))
#     else:
#         test_t1.append(pd.DataFrame([row]))
#         i += 1

training_data = []
testing_data = []
training_labels = []
testing_labels = []

training_data, testing_data, training_labels, testing_labels = train_test_split(T, Gender, train_size=0.5)
print 'traind data : ', len(training_data), 'train lab : ', len(training_labels)
clf = AdaBoostClassifier(n_estimators = 6000)
classed = clf.fit(training_data, training_labels)
scores = cross_val_score(clf, T, Gender, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print 'basic score : ', clf.score(testing_data, testing_labels)
