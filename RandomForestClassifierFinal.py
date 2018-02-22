import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys

path = '/home/avi/Desktop/Reformatted_data.csv'
# The path where all the data is held

data = pd.read_csv(path)
# Using pandas to read all the data in

T1 = []
T2 = []
Volume = []
PatCode = []
SessID = []
BirthAge = []
ScanAge = []
Gender = []

for row in data.itertuples():
    PatCode.append(pd.DataFrame([row[2]]))
    SessID.append(pd.DataFrame([row[3]]))
    BirthAge.append(pd.DataFrame([row[4]]))
    ScanAge.append(pd.DataFrame([row[5]]))
    Gender.append(pd.DataFrame([row[6]]))
    T1.append(pd.DataFrame([row[7:94]]))
    T2.append(pd.DataFrame([row[95:182]]))
    Volume.append(pd.DataFrame([row[183:270]]))
PatCode = pd.concat( PatCode )
SessID = pd.concat( SessID )
BirthAge = pd.concat( BirthAge )
ScanAge = pd.concat( ScanAge )
Gender = pd.concat( Gender )
# PatCode = np.ravel( pd.concat( PatCode ) )
# SessID = np.ravel( pd.concat( SessID ) )
# BirthAge = np.ravel( pd.concat( BirthAge ) )
# ScanAge = np.ravel( pd.concat( ScanAge ) )
# Gender = np.ravel( pd.concat( Gender ) )
T1 = pd.concat(T1)
T2 = pd.concat(T2)
Volume = pd.concat(Volume)

train_t1 = []
test_t1 = []
i = 1

for row in T1.itertuples():
    if i % 2 == 0:
        train_t1.append(pd.DataFrame([row]))
    else:
        test_t1.append(pd.DataFrame([row]))
        i += 1
train_t1 = pd.concat(train_t1)
test_t1 = pd.concat(test_t1)


        train_lab.append(pd.DataFrame([]))


training_data = train_t1
training_labels = Gender

# Features : GA Birth, GA Scan, Gender, Region, T1 intensity, T2 intensity, volume
# Max features = <sqrt(7) = 2
# Max depth : 3,5,7,9 - compare
# No_estimators : 10, 30, 100 - compare (generall the more the bettwer however the cost of
# learning increases and the benefit of learning decreases as you go up, 100 probably
# too many, maybe try 50 or 60 as well and compare)

clf = RandomForestClassifier(max_depth = 3, max_features = 2, n_estimators = 10,
                             max_leaf_nodes = 30)
clf = clf.fit(training_data, training_labels)

clf.predict(test_t1)
# print clf.feature_importances_






























#
