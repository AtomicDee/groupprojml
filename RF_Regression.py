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

# path = '/home/avi/Desktop/groupprojml/DATA/All_Patients'
path = '/home/avi/Desktop/Final_Data_groupprojml/' #for all data
all_data_path = path + 'term+preterm/'
preterm_path = path + 'preterm_below_37/'
term_path = path + 'Term_Data/'

all_data_all_features = pd.read_csv(all_data_path + 'ALL_FEATURES.csv')
all_data_all_labels = pd.read_csv(all_data_path + 'Term_labels.csv')
all_data_all_BirthGA = pd.read_csv(all_data_path + 'BirthGA.csv')

preterm_all_features = pd.read_csv(preterm_path + 'ALL_FEATURES.csv')
preterm_all_labels = pd.read_csv(preterm_path + 'Term_labels.csv')
preterm_all_BirthGA = pd.read_csv(preterm_path + 'BirthGA.csv')

term_all_features = pd.read_csv(term_path + 'ALL_FEATURES.csv')
term_all_labels = pd.read_csv(term_path + 'Term_labels.csv')
term_all_BirthGA = pd.read_csv(term_path + 'BirthGA.csv')

# ScanAge = pd.read_csv(path + '/ScanGA.csv')
# BirthAge = pd.read_csv(path + '/BirthGA.csv')
# all_feat = pd.read_csv(path + '/T1_T2_Vol.csv')

# path1 = '/home/avi/Desktop/test_data_only/Term_Data'
# path2 = '/home/avi/Desktop/test_data_only/Preterm_Data'
# term_ScanAge = pd.read_csv(path1 + '/ScanGA.csv')
# term_BirthAge = pd.read_csv(path1 + '/BirthGA.csv')
# term_all_feat = pd.read_csv(path1 + '/T1_T2_Vol.csv')
# term_labels = pd.read_csv(path1 + '/Term_labels.csv')
# preterm_ScanAge = pd.read_csv(path2 + '/ScanGA.csv')
# preterm_BirthAge = pd.read_csv(path2 + '/BirthGA.csv')
# preterm_all_feat = pd.read_csv(path2 + '/T1_T2_Vol.csv')
# preterm_labels = pd.read_csv(path2 + '/Term_labels.csv')

# the problem with setting some parameters too high is that it becomes overfit
# to the training data, so that when being run the clf becomes crappy at fitting
# test data

depth = [3, 5, 7, 9, 11, 15, 17, 19, 21] # 11(.score()) # 15(cross_val_score())
estim = [30, 50, 100, 300, 500, 800, 1000] # 100(.score()) # 300(cross_val_score())
training_size = [0.5, 0.7, 0.8, 0.9]
# training_size = [0.9]

curr_max = -10000
curr_max_std = -10000
all_best_sizes = []
all_best_depths = []
all_best_estims = []
all_best_scores = []
all_best_cross_scores = []
all_best_std = []
all_best_oob = []
max_oob = 1

testing_data = preterm_all_feat
testing_labels = preterm_ScanAge
training_data = term_all_feat
training_labels = term_ScanAge
testing_data = testing_data[testing_data.columns[1:]]
testing_labels = np.ravel(testing_labels[testing_labels.columns[1:]])
training_data = training_data[training_data.columns[1:]]
training_labels = np.ravel(training_labels[training_labels.columns[1:]])

for x in range(1):
    for k in range(len(training_size)):
        # training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size[k])
        # testing_data = testing_data[testing_data.columns[1:]]
        # testing_labels = np.ravel(testing_labels[testing_labels.columns[1:]])
        # training_data = training_data[training_data.columns[1:]]
        # training_labels = np.ravel(training_labels[training_labels.columns[1:]])
        for i in range(len(depth)):
            curr_score = []
            for j in range(len(estim)):
                print ' '
                print 'Loop: ', x
                print 'Training Size: ', training_size[k]
                print 'Depth: ', depth[i]
                print 'Estimator: ', estim[j]
                regr = RandomForestRegressor(max_depth = depth[i], max_features = 16,
                                            n_estimators = estim[j], oob_score=True)
                regr.fit(training_data, training_labels)
                pred = regr.predict(testing_data)
                # curr_score = regr.score(testing_data, pred)
                # curr_cross_score = cross_val_score(regr, testing_data, pred)
                # curr_mean = np.mean(curr_cross_score)
                # print(curr_score)
                # print(curr_mean)

                curr_score = regr.score(testing_data, testing_labels)
                curr_cross_score = cross_val_score(regr, testing_data, testing_labels)
                curr_mean = np.mean(curr_cross_score)
                # print(curr_score)
                # print(curr_mean)
                # raw_input()
                # d:5, e:500, s:0.80

# ########################################################################
# The best cross score found was:  0.977833218642
# With a STD of:  0.00411351227343
# The best score is:  1.0
# The optimal depth found was:  11.0
# The optimal number of estimators was found to be:  650.0
# The best training size is:  0.65
# The best OOB score is:  0.763446835366
# ########################################################################
#
# ########################################################################
# All cross scores:  [0.9777713284590147, 0.9778951088252797]
# All STDs:  [0.003220300839356766, 0.005006723707494586]
# All scores:  [1.0, 1.0]
# All depths:  [5, 17]
# All estimators:  [1000, 300]
# All training sizes:  [0.5, 0.8]
# All OOB scores:  [0.7559621469297985, 0.7709315238023038]
# #######################################################################


                curr_std = np.std(curr_cross_score)
                curr_oob = regr.oob_score_
                # if curr_oob < max_oob and curr_mean > curr_max:
                if curr_score > curr_max:
                    max_oob = curr_oob
                    curr_cross_max = curr_mean
                    curr_max = curr_score
                    curr_score_max = curr_score
                    curr_max_std = curr_std
                    best_depth = depth[i]
                    best_estim = estim[j]
                    best_size = training_size[k]
                    print 'OOB: ', max_oob
                    print 'Curr Score: ', curr_max
    all_best_sizes.append(best_size)
    all_best_depths.append(best_depth)
    all_best_estims.append(best_estim)
    all_best_scores.append(curr_score_max)
    all_best_cross_scores.append(curr_cross_max)
    all_best_std.append(curr_max_std)
    all_best_oob.append(max_oob)
    curr_cross_max = 0
    curr_score_max
    curr_max = 0
    curr_max_std = 0
    best_depth = 0
    best_estim = 0
    best_size = 0
    max_oob = 0

print ' '
print '########################################################################'
print 'The best cross score found was: ', np.mean(all_best_cross_scores)
print 'With a STD of: ', np.mean(all_best_std)
print 'The best score is: ', np.mean(all_best_scores)
print 'The optimal depth found was: ', np.mean(all_best_depths)
print 'The optimal number of estimators was found to be: ', np.mean(all_best_estims)
print 'The best training size is: ', np.mean(all_best_sizes)
print 'The best OOB score is: ', np.mean(all_best_oob)
print '########################################################################'
print ' '
print '########################################################################'
print 'All cross scores: ', all_best_cross_scores
print 'All STDs: ', all_best_std
print 'All scores: ', all_best_scores
print 'All depths: ', all_best_depths
print 'All estimators: ', all_best_estims
print 'All training sizes: ', all_best_sizes
print 'All OOB scores: ', all_best_oob
print '########################################################################'
print ' '

tim = time.time() - start_time
print 'This took', tim, 'seconds to run.'
print 'Or', tim/60, 'minutes to run'
print 'Or', tim/3600, 'hours to run'
os.system('play --no-show-progress --null -- channels 1 synth %s sin %f' % (1, 550))
