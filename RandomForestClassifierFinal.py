import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
PatCode = np.ravel( pd.concat( PatCode ) )
SessID = np.ravel( pd.concat( SessID ) )
BirthAge = np.ravel( pd.concat( BirthAge ) )
ScanAge = np.ravel( pd.concat( ScanAge ) )
Gender = np.ravel( pd.concat( Gender ) )
T1 = pd.concat(T1)
T2 = pd.concat(T2)
Volume = pd.concat(Volume)

training_data = []
testing_data = []
training_labels = []
testing_labels = []

seed = np.random.seed(42)
training_data, testing_data, training_labels, testing_labels = train_test_split(T1, Gender, train_size=0.5)
# the problem with setting some parameters too high is that it becomes overfit
# to the training data, so that when being run the clf becomes crappy at fitting
# test data
clf = RandomForestClassifier(max_depth = 9, max_features = 2, n_estimators = 500,
                             max_leaf_nodes = 100)
clf.fit(training_data, training_labels)
pred = clf.predict(testing_data, testing_labels)
print clf.score(testing_data, testing_labels)
print accuracy_score()
# print Gender.size
