import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

path = '/home/avi/Desktop/groupprojml/DATA/All_Patients'
ScanAge = pd.read_csv(path + '/ScanGA.csv')
BirthAge = pd.read_csv(path + '/BirthGA.csv')
all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')
Term_labels = pd.read_csv(path + '/Term_labels.csv')

best_depth = 11 # 13.4 depth OOB
best_estim = 136
training_size = 0.77 # 0.53 training size OOB
testing_size = 1 - training_size

# The best cross score found was:  0.907863716644
# With a STD of:  0.0391302667515
# The best score is:  0.885818891775
# The optimal depth found was:  11.4
# The optimal number of estimators was found to be:  136.0
# The best training size is:  0.766666666667
# The best OOB score is:  0.861512398013

score = 0.82
all_scores = []
best_clf = []
best_fit = []
best_pred = []
best_train_data = []
best_test_data = []
best_train_labels = []
best_test_labels = []
for i in range(10):
    training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, Term_labels, train_size=training_size, test_size=testing_size)
    testing_data = testing_data[testing_data.columns[1:]]
    testing_labels = np.ravel(testing_labels[testing_labels.columns[1:]])
    training_data = training_data[training_data.columns[1:]]
    training_labels = np.ravel(training_labels[training_labels.columns[1:]])

    clf_opt = RandomForestClassifier(max_depth = best_depth, max_features = 16, n_estimators = best_estim, oob_score=True)
    clf_opt.fit(training_data, training_labels)

    pred = clf_opt.predict(testing_data)

    curr_cross_score = cross_val_score(clf_opt, testing_data, testing_labels)
    curr_mean = np.mean(curr_cross_score)
    curr_std = np.std(curr_cross_score)
    curr_score = clf_opt.score(testing_data, testing_labels)

    all_scores.append(curr_mean)
    if curr_mean > score:
        score = curr_mean
        best_clf = RandomForestClassifier(max_depth = best_depth, max_features = 16, n_estimators = best_estim, oob_score=True)
        best_clf.fit(training_data, training_labels)
        best_pred = pred
        best_train_data = training_data
        best_test_data = testing_data
        best_train_labels = training_labels
        best_test_labels = testing_labels
    else:
        continue

curr_cross_score = cross_val_score(best_clf, best_test_data, best_test_labels)
curr_mean = np.mean(curr_cross_score)
curr_std = np.std(curr_cross_score)
curr_score = clf_opt.score(best_test_data, best_test_labels)

print ' '
print 'Cross-Validation score for this run:', curr_mean
print 'Cross-Validation STD for this run:', curr_std
print 'Score for this run:', curr_score
print ' '

feat_imp = best_clf.feature_importances_

T1_feature_scores = []
T2_feature_scores = []
Vol_feature_scores = []

T1_feature_scores.append(pd.DataFrame(feat_imp[:86]))
T2_feature_scores.append(pd.DataFrame(feat_imp[86:172]))
Vol_feature_scores.append(pd.DataFrame(feat_imp[172:]))

T1_feature_scores = pd.concat(T1_feature_scores)
T2_feature_scores = pd.concat(T2_feature_scores)
Vol_feature_scores = pd.concat(Vol_feature_scores)

T1_feature_scores.to_csv('T1_features_RF_Clf.csv')
T2_feature_scores.to_csv('T2_features_RF_Clf.csv')
Vol_feature_scores.to_csv('Vol_features_RF_Clf.csv')

# import os
# import numpy as np
# import pandas as pd
# from sklearn import tree
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
#
# path = '/home/avi/Desktop/groupprojml/DATA/All_Patients'
# ScanAge = pd.read_csv(path + '/ScanGA.csv')
# BirthAge = pd.read_csv(path + '/BirthGA.csv')
# all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')
# Term_labels = pd.read_csv(path + '/Term_labels.csv')
#
# best_depth = 11 # 13.4 depth OOB
# best_estim = 136
# training_size = 0.77 # 0.53 training size OOB
# testing_size = 1 - training_size
#
# # The best cross score found was:  0.907863716644
# # With a STD of:  0.0391302667515
# # The best score is:  0.885818891775
# # The optimal depth found was:  11.4
# # The optimal number of estimators was found to be:  136.0
# # The best training size is:  0.766666666667
# # The best OOB score is:  0.861512398013
#
# training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, Term_labels, train_size=training_size, test_size=testing_size)
# testing_data = testing_data[testing_data.columns[1:]]
# testing_labels = np.ravel(testing_labels[testing_labels.columns[1:]])
# training_data = training_data[training_data.columns[1:]]
# training_labels = np.ravel(training_labels[training_labels.columns[1:]])
#
# clf_opt = RandomForestClassifier(max_depth = best_depth, max_features = 16, n_estimators = best_estim, oob_score=True)
# clf_opt.fit(training_data, training_labels)
#
# pred = clf_opt.predict(testing_data)
#
# curr_cross_score = cross_val_score(clf_opt, testing_data, testing_labels)
# curr_mean = np.mean(curr_cross_score)
# curr_std = np.std(curr_cross_score)
# curr_score = clf_opt.score(testing_data, testing_labels)
#
# print ' '
# print 'Cross-Validation score for this run:', curr_mean
# print 'Cross-Validation STD for this run:', curr_std
# print 'Score for this run:', curr_score
# # print 'OOB score: ', regr_opt.oob_score_
# print ' '
#
# feat_imp = clf_opt.feature_importances_
# T1_feature_scores = []
# T2_feature_scores = []
# Vol_feature_scores = []
#
# T1_feature_scores.append(pd.DataFrame(feat_imp[:86]))
# T2_feature_scores.append(pd.DataFrame(feat_imp[86:172]))
# Vol_feature_scores.append(pd.DataFrame(feat_imp[172:]))
#
# T1_feature_scores = pd.concat(T1_feature_scores)
# T2_feature_scores = pd.concat(T2_feature_scores)
# Vol_feature_scores = pd.concat(Vol_feature_scores)
# # print(len(feat_imp))
# # print(len(T1_feature_scores))
# # print(len(T2_feature_scores))
# # print(len(Vol_feature_scores))
# T1_feature_scores.to_csv('T1_features_RF_Clas.csv')
# T2_feature_scores.to_csv('T2_features_RF_Clas.csv')
# Vol_feature_scores.to_csv('Vol_features_RF_Clas.csv')
#
# # plt.figure(1)
# # plt.scatter(pred, testing_labels, c='b', label='Prediction')
# # plt.plot([0, 50], [0, 50], '--k')
# # plt.xlabel('Data')
# # plt.ylabel('Target')
# # plt.xlim([28, 50])
# # plt.ylim([25, 48])
# # plt.title('Random Forest Classification')
# # plt.legend()
# # plt.show()
