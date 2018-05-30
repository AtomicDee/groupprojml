import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
start_time = time.time()
from sklearn.metrics import accuracy_score

# including all paths
path = '/media/avi/MY USB 3/New folder/GROUP_PROJECT/'
above_37_all_data_path = path + 'term+preterm/'
preterm_path = path + 'preterm_above_37/'
all_data_path = path + 'ALL_PATIENTS_NO_DUPLICATES/'
term_path = path + 'Term_Data/'

# above_37_all_data_all_features = pd.read_csv(above_37_all_data_path + 'ALL_FEATURES.csv')
above_37_all_data_all_labels = pd.read_csv(above_37_all_data_path + 'Term_labels.csv')
above_37_all_data_all_t1t2vol = pd.read_csv(above_37_all_data_path + 'T1_T2_Vol.csv')
above_37_all_data_all_BirthGA = pd.read_csv(above_37_all_data_path + 'BirthGA.csv')
above_37_all_data_all_genders = pd.read_csv(above_37_all_data_path + 'Gender.csv')
above_37_all_data_all_ScanGA = pd.read_csv(above_37_all_data_path + 'ScanGA.csv')

# all_data_all_features = pd.read_csv(all_data_path + 'ALL_FEATURES.csv')
all_data_all_labels = pd.read_csv(all_data_path + 'Term_labels.csv')
all_data_all_t1t2vol = pd.read_csv(all_data_path + 'T1_T2_Vol.csv')
all_data_all_BirthGA = pd.read_csv(all_data_path + 'BirthGA.csv')
all_data_all_genders = pd.read_csv(all_data_path + 'Gender.csv')
all_data_all_ScanGA = pd.read_csv(all_data_path + 'ScanGA.csv')

split = [2, 4, 7]
estim = [10, 30, 100, 300] # 100(.score()) # 300(cross_val_score())
depths = [4, 7, 10, 11]


''' These lines were commented/uncommented based on what was being tested  '''
features = all_data_all_t1t2vol[all_data_all_t1t2vol.columns[1:]]
# labels = all_data_all_genders[all_data_all_genders.columns[2:]]
labels = all_data_all_labels[all_data_all_labels.columns[2:]]
# labels = all_data_all_BirthGA[all_data_all_BirthGA.columns[2:]]
# labels = all_data_all_ScanGA[all_data_all_ScanGA.columns[2:]]
# features = above_37_all_data_all_t1t2vol[above_37_all_data_all_t1t2vol.columns[1:]]
# labels = above_37_all_data_all_genders[above_37_all_data_all_genders.columns[1:]]
# labels = above_37_all_data_all_labels[above_37_all_data_all_labels.columns[1:]]
# labels = above_37_all_data_all_BirthGA[above_37_all_data_all_BirthGA.columns[1:]]
# labels = above_37_all_data_all_ScanGA[above_37_all_data_all_ScanGA.columns[1:]]


# Initialise the final results matrix
all_data = np.zeros([3, 9])
all_data[0, 0] = 0.5
all_data[1, 0] = 0.7
all_data[2, 0] = 0.9
name = 'best_results.csv'
name2 = 'best_results_features.csv'
# setting the random state
RS1 = 42
RS2 = 42
# presetting the inital best score - would be immediately overwritten
best_score = -1000

training_size = [0.5, 0.7, 0.9]
for i in range(len(training_size)):
    # creates the training and testing datasets
    training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=training_size[i], random_state=RS1) #estimating scan age
    # print(training_data)
    # raw_input()
    testing_labels = np.ravel(testing_labels)
    training_labels = np.ravel(training_labels)
    print('size', training_size[i])
    for j in range(len(split)):
        print('split', split[j])
        for k in range(len(depths)):
            print('depth', depths[k])
            for l in range(len(estim)):
                print('estim', estim[l])
                ''' can choose between a classifier and regressor for the method being tested '''
                # model = RandomForestRegressor(max_depth=depths[k], max_features=16, n_estimators=estim[l], min_samples_split=split[j], random_state=RS2)
                model = RandomForestClassifier(max_depth=depths[k], max_features=16, n_estimators=estim[l], min_samples_split=split[j], random_state=RS2)
                model.fit(training_data,training_labels)
                cv_score = cross_val_score(model, testing_data, testing_labels)
                # Testing to see if code works and returns values
                # print(cv_score)
                # cv_score = cross_val_score(model, training_data, training_labels)
                # print(cv_score)
                # cv_score = cross_val_score(model, testing_data, testing_labels)
                # print(cv_score)
                # raw_input()
                # print(np.mean(cv_score))
                if np.mean(cv_score) > best_score:
                    ''' if a best score is found, save the parameters '''
                    print('New Best Score Found')
                    # raw_input()
                    score = model.score(testing_data, testing_labels)
                    best_cv = cv_score
                    best_score = np.mean(cv_score)
                    best_std = np.std(cv_score)
                    best_size = training_size[i]
                    best_split = split[j]
                    best_depth = depths[k]
                    best_estim = estim[l]
                    cv_score = -1
                    print(score)
                    # feat_imp = model.feature_importances_
    ''' save the best parameters for the training size that was just evaluated '''
    all_data[i, 0] = best_size
    all_data[i, 1] = best_score
    all_data[i, 2] = best_std
    all_data[i, 3] = best_estim
    all_data[i, 4] = best_split
    all_data[i, 5] = best_depth
    all_data[i, 6] = RS1
    all_data[i, 7] = RS2
    all_data[i, 8] = score
    print(all_data)
    ''' reset the parameters so that the the next training size can be evaluated properly '''
    best_score = 0
    best_cv = 0
    best_split = 0
    best_estim = 0
    best_size = 0
    best_depth = 0
    best_std = 0
    best_score = -1000
    score = 0

'''
wait for the user to have an input before saving the results - this way multiple
versions of this script can be run at once, testing for different things
(regr/clf, etc)
'''
raw_input()
np.savetxt(name, all_data, delimiter=",")
# np.savetxt(name2, feat_imp, delimiter=',')
# os.system('play --no-show-progress --null -- channels 1 synth %s sin %f' % (1, 550))


print(' ')
print('All')
print(all_data)
print(' ')
