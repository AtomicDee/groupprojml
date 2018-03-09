import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# path = '/home/avi/Desktop/all_normalised_data.csv'
path = '/home/avi/Desktop/groupprojml/GROUP_PROJ_DATA'

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

# the problem with setting some parameters too high is that it becomes overfit
# to the training data, so that when being run the clf becomes crappy at fitting
# test data

depth = [5, 7, 9, 11, 15, 17, 19, 21] # 11(.score()) # 15(cross_val_score())
estim = [30, 50, 100, 300, 500, 800] # 100(.score()) # 300(cross_val_score())
training_size = [0.5, 0.7, 0.8, 0.9]

# depth = [11, 15, 17, 19, 21] # 11(.score()) # 15(cross_val_score())
# estim = [30, 50, 100, 300, 500, 800] # 100(.score()) # 300(cross_val_score())
# depth = [15]
# estim = [50]
table = np.zeros((np.size(depth), np.size(estim)))

# best_depth = 21 #0.9
# best_estim = 30 #0.9
# best_depth = 15 #0.5
# best_estim = 300 #0.5
# best_size = 0
# best_depth = 15 #0.8
# best_estim = 100 #0.8
#
curr_max = -10000
curr_max_std = -10000
# ave = [None] * 5

all_best_sizes = []
all_best_depths = []
all_best_estims = []
all_best_scores = []
all_best_cross_scores = []

############################################################ Training size:  0.9
##################################################################### Depth:  15
################################################################ Estimation:  50
# best_depth = 100
# best_estim = 50
# training_size = 0.9
# training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size)

# testing_data = testing_data[testing_data.columns[1:]]
# testing_labels = testing_labels[testing_labels.columns[1:]]
#
# training_data = training_data[training_data.columns[1:]]
# training_labels = training_labels[training_labels.columns[1:]]

for k in range(len(training_size)):
    training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size[k])
    for i in range(len(depth)):
        curr_score = []
        for j in range(len(estim)):
            regr = RandomForestRegressor(max_depth = depth[i], max_features = 16,
                                        n_estimators = estim[j])
            regr.fit(training_data, training_labels)
            curr_score = regr.score(testing_data, testing_labels)
            curr_cross_score = cross_val_score(regr, testing_data, testing_labels)
            curr_mean = np.mean(curr_cross_score)
            curr_std = np.std(curr_cross_score)
            table[i][j] = curr_mean
            if curr_mean > curr_max:
                curr_cross_max = curr_mean
                curr_max = curr_score
                curr_max_std = curr_std
                best_depth = depth[i]
                best_estim = estim[j]
                best_size = training_size[k]
                print ' '
                print 'Training size: ', training_size[k]
                print 'Depth: ', depth[i]
                print 'Estimation: ', estim[j]
                print 'Cross val score: ', curr_cross_max
                print 'Cross val std: ', curr_max_std
                print 'Score: ', curr_max

print ' '
print 'The best cross score found was: ', curr_cross_max
print 'With a STD of: ', curr_max_std
print 'The best score is: ', curr_max
print 'The optimal depth found was: ', best_depth
print 'The optimal number of estimators was found to be: ', best_estim
print 'The best training size is: ', best_size
# print 'The optimal training size is: ', np.mean(ave)
print table
print ' '

# The best cross score found was:  0.900382049012
# With a STD of:  0.0211345141403
# The best score is:  0.905793153858
# The optimal depth found was:  11
# The optimal number of estimators was found to be:  300
# The best training size is:  0.5
# best cross score found was:  0.911645462341
# With a STD of:  0.0249744785107
# The best score is:  0.925722181604
# The optimal depth found was:  5
# The optimal number of estimators was found to be:  30
# The best training size is:  0.7
# The best cross score found was:  0.871062026774
# With a STD of:  0.0140868768709
# The best score is:  0.910451892723
# The optimal depth found was:  5
# The optimal number of estimators was found to be:  30
# The best training size is:  0.8
# The best cross score found was:  0.882416262025
# With a STD of:  0.0172555872232
# The best score is:  0.901312736719
# The optimal depth found was:  5
# The optimal number of estimators was found to be:  30
# The best training size is:  0.7






# no. features = 261 --> max_features = sqrt(261)~=16

regr_opt = RandomForestRegressor(max_depth = best_depth, max_features = 16,
                                n_estimators = best_estim)
regr_opt.fit(training_data, training_labels)

# Predict
pred = regr_opt.predict(testing_data)

curr_score = cross_val_score(regr_opt, testing_data, testing_labels)
curr_mean = np.mean(curr_score)
curr_std = np.std(curr_score)
feat_imp = regr_opt.feature_importances_

T1_feature_scores = []
T2_feature_scores = []
Vol_feature_scores = []

T1_feature_scores.append(pd.DataFrame(feat_imp[1:88]))
T2_feature_scores.append(pd.DataFrame(feat_imp[88:175]))
Vol_feature_scores.append(pd.DataFrame(feat_imp[175:]))

T1_feature_scores = pd.concat(T1_feature_scores)
T2_feature_scores = pd.concat(T2_feature_scores)
Vol_feature_scores = pd.concat(Vol_feature_scores)

# T1_feature_scores.to_csv('T1_features.csv')
# T2_feature_scores.to_csv('T2_features.csv')
# Vol_feature_scores.to_csv('Vol_features.csv')

print ' '
print 'Cross-Valdation Score: ', curr_mean
print 'With a standard deviation of: ', curr_std
# print ' '
# print pred
# print ' '
# print testing_labels[testing_labels.columns[1:]]
# print type(training_data)
# new_testing_data = testing_data[testing_data.columns[1:]]
# print training_labels
#
# print testing_labels
plt.figure(1)
plt.scatter(pred, testing_labels, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Data')
plt.ylabel('Target')
plt.xlim([28, 45])
plt.ylim([25, 50])
plt.title('Random Forest Regression')
plt.legend()
plt.show()
