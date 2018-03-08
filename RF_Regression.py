import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# path = '/home/avi/Desktop/all_normalised_data.csv'
path = '/home/avi/Desktop/GROUP_PROJ_DATA'

T1 = pd.read_csv(path + '/T1.csv')
T2 = pd.read_csv(path + '/T2.csv')
Gender = pd.read_csv(path + '/Gender.csv')
SessID = pd.read_csv(path + '/SessID.csv')
Volume = pd.read_csv(path + '/Volume.csv')
ScanAge = pd.read_csv(path + '/ScanGA.csv')
PatCode = pd.read_csv(path + '/PatCode.csv')
BirthAge = pd.read_csv(path + '/BirthGA.csv')
all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')

testing_data = []
training_data = []
testing_labels = []
training_labels = []

training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=0.5)
# the problem with setting some parameters too high is that it becomes overfit
# to the training data, so that when being run the clf becomes crappy at fitting
# test data

# depth = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21] # 11(.score()) # 15(cross_val_score())
# estim = [10, 30, 50, 100, 300, 500, 800, 1000] # 100(.score()) # 300(cross_val_score())
# table = np.zeros((np.size(depth), np.size(estim)))
#
# best_depth = 0
# best_estim = 0
#
# best_depth = 21 #0.9
# best_estim = 30 #0.9
best_depth = 15 #0.5
best_estim = 300 #0.5
# best_depth = 15 #0.8
# best_estim = 100 #0.8
#
# curr_max = -10000
# curr_max_std = -10000
#
# for i in range(len(depth)):
#     curr_score = []
#     for j in range(len(estim)):
#         regr = RandomForestRegressor(max_depth = depth[i], max_features = 16,
#                                     n_estimators = estim[j])
#         regr.fit(training_data, training_labels)
#         # curr_score = clf.score(testing_data, testing_labels)
#         curr_score = cross_val_score(regr, testing_data, testing_labels)
#         curr_mean = np.mean(curr_score)
#         curr_std = np.std(curr_score)
#         # raw_input()
#         table[i][j] = curr_mean
#         if curr_mean > curr_max:
#             curr_max = curr_mean
#             curr_max_std = curr_std
#             best_depth = depth[i]
#             best_estim = estim[j]
# print ' '
# print 'The best accuracy score found was: ', curr_max
# print 'The optimal depth found was: ', best_depth
# print 'The optimal number of estimators was found to be: ', best_estim
# print ' '
# # print table
#
# raw_input()
#
# no. features = 261 --> max_features = sqrt(261)~=16

regr_opt = RandomForestRegressor(max_depth = best_depth, max_features = 16,
                                n_estimators = best_estim)
regr_opt.fit(training_data, training_labels)

# Predict
train_pred = regr_opt.predict(testing_data)
pred = regr_opt.predict(testing_data)

curr_score = cross_val_score(regr_opt, testing_data, testing_labels)
curr_mean = np.mean(curr_score)
curr_std = np.std(curr_score)
feat_imp = regr_opt.feature_importances_

T1_feature_scores = feat_imp[1:88]
T2_feature_scores = feat_imp[88:175]
Vol_feature_scores = feat_imp[175:]

print ' '
print 'Cross-Valdation Score: ', curr_mean
print 'With a standard deviation of: ', curr_std
print type(feat_imp)
print type(T1_feature_scores)
print ' '

np.savetxt('T1_features.csv', T1_feature_scores, delimiter=",")
np.savetxt('T2_features.csv', T2_feature_scores, delimiter=",")
np.savetxt('Vol_features.csv', Vol_feature_scores, delimiter=",")
# T1_feature_scores.to_csv('T1_features.csv')
# T2_feature_scores.to_csv('T2_features.csv')
# Vol_feature_scores.to_csv('Vol_features.csv')


# plt.figure(1)
# plt.scatter(pred, testing_labels, c="k", label='Prediction')
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Random Forest Regression")
# plt.legend()
# plt.show()
#
# plt.figure(2)
# plt.scatter(train_pred, training_labels, c='g', label ='Training Samples')
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Random Forest Regression")
# plt.legend()
# plt.show()
