import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

path = '/home/avi/Desktop/groupprojml/DATA/All_Patients'
ScanAge = pd.read_csv(path + '/ScanGA.csv')
BirthAge = pd.read_csv(path + '/BirthGA.csv')
all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')

best_depth = 9 # 8.5 depth
best_estim = 240 # 240 Estims
training_size = 0.62 # 0.62 training size
testing_size = 1 - training_size

training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size, test_size=testing_size)
testing_data = testing_data[testing_data.columns[1:]]
testing_labels = np.ravel(testing_labels[testing_labels.columns[1:]])
training_data = training_data[training_data.columns[1:]]
training_labels = np.ravel(training_labels[training_labels.columns[1:]])

regr_opt = RandomForestRegressor(max_depth = best_depth, max_features = 16, n_estimators = best_estim, oob_score=True)
regr_opt.fit(training_data, training_labels)

pred = regr_opt.predict(testing_data)

curr_cross_score = cross_val_score(regr_opt, testing_data, testing_labels)
curr_mean = np.mean(curr_cross_score)
curr_std = np.std(curr_cross_score)
curr_score = regr_opt.score(testing_data, testing_labels)

print ' '
print 'Cross-Validation score for this run:', curr_mean
print 'Cross-Validation STD for this run:', curr_std
print 'Score for this run:', curr_score
print 'OOB score: ', regr_opt.oob_score_
print ' '

feat_imp = regr_opt.feature_importances_
T1_feature_scores = []
T2_feature_scores = []
Vol_feature_scores = []

T1_feature_scores.append(pd.DataFrame(feat_imp[:86]))
T2_feature_scores.append(pd.DataFrame(feat_imp[86:172]))
Vol_feature_scores.append(pd.DataFrame(feat_imp[172:]))

T1_feature_scores = pd.concat(T1_feature_scores)
T2_feature_scores = pd.concat(T2_feature_scores)
Vol_feature_scores = pd.concat(Vol_feature_scores)
# T1_feature_scores.to_csv('T1_features.csv')
# T2_feature_scores.to_csv('T2_features.csv')
# Vol_feature_scores.to_csv('Vol_features.csv')

plt.figure(1)
plt.scatter(pred, testing_labels, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Data')
plt.ylabel('Target')
plt.xlim([28, 50])
plt.ylim([25, 48])
plt.title('Random Forest Regression')
plt.legend()
plt.show()
