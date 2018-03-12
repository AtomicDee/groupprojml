import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
start_time = time.time()

path = '/home/avi/Desktop/groupprojml/DATA/All_Patients'

ScanAge = pd.read_csv(path + '/ScanGA.csv')
BirthAge = pd.read_csv(path + '/BirthGA.csv')
all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')

# the problem with setting some parameters too high is that it becomes overfit
# to the training data, so that when being run the clf becomes crappy at fitting
# test data

depth = [3, 5, 7, 9, 11, 15, 17, 19, 21] # 11(.score()) # 15(cross_val_score())
estim = [30, 50, 100, 300, 500, 800, 1000] # 100(.score()) # 300(cross_val_score())
training_size = [0.5, 0.7, 0.8, 0.9]

curr_max = -10000
curr_max_std = -10000
all_best_sizes = []
all_best_depths = []
all_best_estims = []
all_best_scores = []
all_best_cross_scores = []
all_best_std = []

for x in range(30):
    for k in range(len(training_size)):
        training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size[k])
        for i in range(len(depth)):
            curr_score = []
            for j in range(len(estim)):
                print ' '
                print 'Loop: ', x
                print 'Training Size: ', training_size[k]
                print 'Depth: ', depth[i]
                print 'Estimator: ', estim[j]
                regr = RandomForestRegressor(max_depth = depth[i], max_features = 16,
                                            n_estimators = estim[j])
                regr.fit(training_data, training_labels)
                curr_score = regr.score(testing_data, testing_labels)
                curr_cross_score = cross_val_score(regr, testing_data, testing_labels)
                curr_mean = np.mean(curr_cross_score)
                curr_std = np.std(curr_cross_score)
                if curr_mean > curr_max:
                    curr_cross_max = curr_mean
                    curr_max = curr_score
                    curr_max_std = curr_std
                    best_depth = depth[i]
                    best_estim = estim[j]
                    best_size = training_size[k]
    # print ' '
    # print 'The best cross score found was: ', curr_cross_max
    # print 'With a STD of: ', curr_max_std
    # print 'The best score is: ', curr_max
    # print 'The optimal depth found was: ', best_depth
    # print 'The optimal number of estimators was found to be: ', best_estim
    # print 'The best training size is: ', best_size
    # print ' '
    all_best_sizes.append(best_size)
    all_best_depths.append(best_depth)
    all_best_estims.append(best_estim)
    all_best_scores.append(curr_max)
    all_best_cross_scores.append(curr_cross_max)
    all_best_std.append(curr_max_std)
    curr_cross_max = 0
    curr_max = 0
    curr_max_std = 0
    best_depth = 0
    best_estim = 0
    best_size = 0

print ' '
print '########################################################################'
print 'The best cross score found was: ', np.mean(all_best_cross_scores)
print 'With a STD of: ', np.mean(all_best_std)
print 'The best score is: ', np.mean(all_best_scores)
print 'The optimal depth found was: ', np.mean(all_best_depths)
print 'The optimal number of estimators was found to be: ', np.mean(all_best_estims)
print 'The best training size is: ', np.mean(all_best_sizes)
print '########################################################################'
print ' '
print ' '
print '########################################################################'
print 'All cross scores: ', all_best_cross_scores
print 'All STDs: ', all_best_std
print 'All scores: ', all_best_scores
print 'All depths: ', all_best_depths
print 'All estimators: ', all_best_estims
print 'All training sizes: ', all_best_sizes
print '########################################################################'
print ' '

#
# best_depth = np.mean(all_best_depths)
# best_estim = np.mean(all_best_estims)
# training_size = np.mean(all_best_sizes)
# training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size)
# testing_data = testing_data[testing_data.columns[1:]]
# testing_labels = testing_labels[testing_labels.columns[1:]]
# training_data = training_data[training_data.columns[1:]]
# training_labels = training_labels[training_labels.columns[1:]]
#
# regr_opt = RandomForestRegressor(max_depth = best_depth, max_features = 16,
#                                 n_estimators = best_estim)
# regr_opt.fit(training_data, training_labels)
#
# pred = regr_opt.predict(testing_data)
#
# curr_score = cross_val_score(regr_opt, testing_data, testing_labels)
# curr_mean = np.mean(curr_score)
# curr_std = np.std(curr_score)
# feat_imp = regr_opt.feature_importances_
#
# T1_feature_scores = []
# T2_feature_scores = []
# Vol_feature_scores = []
#
# T1_feature_scores.append(pd.DataFrame(feat_imp[1:88]))
# T2_feature_scores.append(pd.DataFrame(feat_imp[88:175]))
# Vol_feature_scores.append(pd.DataFrame(feat_imp[175:]))
#
# T1_feature_scores = pd.concat(T1_feature_scores)
# T2_feature_scores = pd.concat(T2_feature_scores)
# Vol_feature_scores = pd.concat(Vol_feature_scores)
#
# # T1_feature_scores.to_csv('T1_features.csv')
# # T2_feature_scores.to_csv('T2_features.csv')
# # Vol_feature_scores.to_csv('Vol_features.csv')
#
# print ' '
# print 'Cross-Valdation Score: ', curr_mean
# print 'With a standard deviation of: ', curr_std
tim = time.time() - start_time
print 'This took', tim, 'seconds to run.'
print 'Or', tim/60, 'minutes to run'
print 'Or', tim/3600, 'hours to run'
#
# plt.figure(1)
# plt.scatter(pred, testing_labels, c='b', label='Prediction')
# plt.plot([0, 50], [0, 50], '--k')
# plt.xlabel('Data')
# plt.ylabel('Target')
# plt.xlim([28, 45])
# plt.ylim([25, 50])
# plt.title('Random Forest Regression')
# plt.legend()
# plt.show()
