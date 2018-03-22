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
# training_size = [0.5, 0.7, 0.8, 0.9]
training_size = [0.9]

curr_max = -10000
curr_max_std = -10000
all_best_sizes = []
all_best_depths = []
all_best_estims = []
all_best_scores = []
all_best_cross_scores = []
all_best_std = []
all_best_oob = []
max_oob = -10000

for x in range(15):
    for k in range(len(training_size)):
        training_data, testing_data, training_labels, testing_labels = train_test_split(all_feat, ScanAge, train_size=training_size[k])
        testing_data = testing_data[testing_data.columns[1:]]
        testing_labels = np.ravel(testing_labels[testing_labels.columns[1:]])
        training_data = training_data[training_data.columns[1:]]
        training_labels = np.ravel(training_labels[training_labels.columns[1:]])
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
                curr_score = regr.score(testing_data, testing_labels)
                curr_cross_score = cross_val_score(regr, testing_data, testing_labels)
                curr_mean = np.mean(curr_cross_score)
                curr_std = np.std(curr_cross_score)
                curr_oob = regr.oob_score_
                if curr_oob > max_oob and curr_mean > curr_max:
                    max_oob = curr_oob
                    curr_cross_max = curr_mean
                    curr_max = curr_mean
                    curr_score_max = curr_score
                    curr_max_std = curr_std
                    best_depth = depth[i]
                    best_estim = estim[j]
                    best_size = training_size[k]
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
