import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
# import nibabel
# import csv
# import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# path = '/home/avi/Desktop/Reformatted_data.csv'
# path = '/home/avi/Desktop/all_normalised_data.csv'
path = '/home/avi/Desktop/GROUP_PROJ_DATA'
# The path where all the data is held

T1 = pd.read_csv(path + '/T1.csv')
T2 = pd.read_csv(path + '/T2.csv')
Volume = pd.read_csv(path + '/Volume.csv')
PatCode = pd.read_csv(path + '/PatCode.csv')
SessID = pd.read_csv(path + '/SessID.csv')
BirthAge = pd.read_csv(path + '/BirthGA.csv')
ScanAge = pd.read_csv(path + '/ScanGA.csv')
Gender = pd.read_csv(path + '/Gender.csv')
all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')

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
#
# for i in range(len(depth)):
#     curr_score = []
#     for j in range(len(estim)):
#         clf = RandomForestRegressor(max_depth = depth[i], max_features = 16,
#                                     n_estimators = estim[j])
#         clf.fit(training_data, training_labels)
#         prediction = clf.predict(testing_data)
#         curr_score = clf.score(prediction, testing_labels)
#         table[i][j] = curr_score
#         if curr_score > curr_max:
#             curr_max = curr_score
#             best_depth = depth[i]
#             best_estim = estim[i]
# print ' '
# print 'The best accuracy score found was: ', curr_max
# print 'The optimal depth found was: ', best_depth
# print 'The optimal number of estimators was found to be: ', best_estim
# print ' '
# # print table
#
# # no. features = 261 --> max_features = sqrt(261)~=16
# clf_opt = RandomForestRegressor(max_depth = best_depth, max_features = 16,
#                                 n_estimators = best_estim)
# clf_opt.fit(training_data, training_labels)
#
# # Predict
# train_pred = clf_opt.predict(testing_data)
# pred = clf_opt.predict(testing_data)
#
# print np.shape(pred)
# print np.shape(testing_labels)
# print 'ACCURACY SCORE :', accuracy_score(pred, testing_labels)
# print ' '
#
# # print np.shape(training_data)
# # print np.shape(testing_data)
# # print np.shape(training_labels)
# # print np.shape(testing_labels)
# # print np.shape(pred)
# # print np.shape(train_pred)
#
# plt.figure(1)
# plt.scatter(pred, testing_labels, c="k", label='Prediction')
# # plt.scatter(pred, training_labels, c='g', label ='Training Samples')
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Random Forest Regression")
# plt.legend()
# # plt.show()
#
# plt.figure(2)
# # plt.scatter(pred, testing_labels, c="k", label='Prediction')
# plt.scatter(train_pred, training_labels, c='g', label ='Training Samples')
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Random Forest Regression")
# plt.legend()
# plt.show()

















#
