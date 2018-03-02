import pandas as pd
import numpy as np
import os
# from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
# import nibabel
# import csv
# import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path = '/home/avi/Desktop/Reformatted_data.csv'
import matplotlib.pyplot as plt
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
training_data, testing_data, training_labels, testing_labels = train_test_split(T1, ScanAge, train_size=0.5, test_size=0.5)
# the problem with setting some parameters too high is that it becomes overfit
# to the training data, so that when being run the clf becomes crappy at fitting
# test data

depth = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21]
estim = [10, 30, 50, 100, 300, 500, 800, 1000]
table = np.zeros((np.size(depth), np.size(estim)))

best_depth = 0
best_estim = 0
curr_max = -10000

for i in range(len(depth)):
    curr_score = []
    for j in range(len(estim)):
        clf = RandomForestRegressor(max_depth = depth[i], max_features = 2,
                                    n_estimators = estim[j])
        clf.fit(training_data, training_labels)
        curr_score = clf.score(testing_data, testing_labels)
        table[i][j] = curr_score
        if curr_score > curr_max:
            curr_max = curr_score
            best_depth = depth[i]
            best_estim = estim[i]
# print table
print ' '
print 'The best accuracy score found was: ', curr_max
print 'The optimal depth found was: ', best_depth
print 'The optimal number of estimators was found to be: ', best_estim
print ' '
# print table

clf_opt = RandomForestRegressor(max_depth = best_depth, max_features = 2,
                                n_estimators = best_estim)
clf_opt.fit(training_data, training_labels)

# Predict
pred = clf_opt.predict(testing_data)


#
# clf_opt.fit(pred.reshape(1, -1), training_labels)
# pred.score = clf_opt.score(pred.reshape(-1,1), testing_labels)
# # print 'The prediction based on the training data is: ', pred
#
print np.shape(training_data)
print np.shape(testing_data)
print np.shape(training_labels)
print np.shape(testing_labels)
print np.shape(pred)
print np.shape(T1)
#
plt.figure()
plt.scatter(pred, testing_labels, c="k", label='Prediction')
plt.scatter(pred, training_labels, c='g', label ='Training Samples')
# plt.plot(testing_data, pred, c="g", label="n_estimators=1", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()
#
#
#
#
#
#
#
#
#
# #
