import pandas as pd
import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import csv
import os
####################################################


# import some data to play with
path = '/Users/Mickey/Desktop/Group/Term_Data/'

all_Feat = pd.read_csv(path+'T1_T2_Vol_SA_termandpreterm.csv')
#all_Feat = all_Feat[all_Feat.columns[1:]]

labels = pd.read_csv(path+'BirthGA_termandpreterm.csv')
#labels = labels[labels.columns[2]]

########################################################

#x_all = data_im
x_all = all_Feat
y_all = labels

print np.shape(x_all)
print np.shape(y_all)

########################################################
#TRAINING AND TESTING DATA :
# X - Trainin data
# X_test - testing data
# y - traininglabels
# y_test - testing labels
#X, X_test, y, y_test = train_test_split(T1, y_all,train_size=0.9,random_state=42)
All_R = []
ultimate_score = -1000
for count in range(10):
    R1 = np.random.randint(low = 1, high =200)
    All_R.append(R1)
    X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.9,random_state = R1)
    print '0.9'
    y = np.ravel(y)
    ###############################################################################

    gamma = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2,1e3]
    kernel = ['rbf','linear']
    C = [1, 10, 100,1000]

    curr_max = -10000
    curr_min = 10000
    all_gamma = []
    all_C = []
    all_kernel = []
    all_scores = []
    all_cscores = []
    all_std = []



    for x in range(len(kernel)):
         for j in range(len(C)):
            if kernel[x] == 'linear':
                if C[j] < 1000:

                    svm = SVR(kernel= kernel[x], C=C[j])
                    svm.fit(X,y)
                    pre = svm.predict(X_test)

                    curr_score = svm.score(X_test, y_test)
                    curr_cross_score = cross_val_score(svm, X_test, y_test, cv=5)
                    curr_mean = np.mean(curr_cross_score)
                    curr_std = np.std(curr_cross_score)

                    all_scores.append(curr_score)
                    all_cscores.append(curr_mean)
                    all_std.append(curr_std)

                    if curr_score > curr_max:
                        curr_max = curr_score

                        best_cross = curr_mean
                        best_score = curr_score
                        best_std = curr_std
                        best_C = C[j]
                        best_gamma = []
                        best_kernel = kernel[x]

                    if curr_score < curr_min:
                        curr_min = curr_score

                        worst_cross = curr_mean
                        worst_score = curr_score
                        worst_std = curr_std
                        worst_C = C[j]
                        worst_gamma = []
                        worst_kernel = kernel[x]

            if kernel[x] == 'rbf':
                for i in range(len(gamma)):
                    svm = SVR(kernel='rbf', C=C[j],gamma=gamma[i])
                    svm.fit(X,y)
                    pre = svm.predict(X_test)

                    curr_score = svm.score(X_test, y_test)
                    curr_cross_score = cross_val_score(svm, X_test, y_test, cv=5)
                    curr_mean = np.mean(curr_cross_score)
                    curr_std = np.std(curr_cross_score)
                    curr_mean = np.mean(curr_cross_score)
                    curr_std = np.std(curr_cross_score)

                    all_scores.append(curr_score)
                    all_cscores.append(curr_mean)
                    all_std.append(curr_std)

                    if curr_score > curr_max:
                        best_cross = curr_mean
                        curr_max = curr_score

                        best_score = curr_score
                        best_std = curr_std
                        best_C = C[j]
                        best_gamma = gamma[i]
                        best_kernel = kernel[x]

                    if curr_score < curr_min:
                        curr_min = curr_score

                        worst_cross = curr_mean
                        worst_score = curr_score
                        worst_std = curr_std
                        worst_C = C[j]
                        worst_gamma = gamma[i]
                        worst_kernel = kernel[x]

    if best_score > ultimate_score:
        ultimate_score = best_score
        ultimate_cross = best_cross
        ultimate_std = best_std
        ultimate_C = best_C
        ultimate_gamma = best_gamma
        ultimate_kernel = best_kernel
        ultimate_R = R1

print "BEST of the BEST : "
print ultimate_R
print ultimate_score
print ultimate_cross
print ultimate_std
print ultimate_C
print ultimate_gamma
print ultimate_kernel

###############################################################################
