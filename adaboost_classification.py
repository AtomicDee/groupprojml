# AdaBoost classification - use for feature extraction then svm
import pandas as pd
import nibabel
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile

from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# X, y = make_gaussian_quantiles(n_samples=1000, n_features=10,
#                                n_classes=3, random_state=1)
path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/split_data/term_preterm/"
path_save = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/class_gen_red_results/"
T1 = pd.read_csv(path+'T1.csv')
print 'Shape T1 : ', T1.shape
T2 = pd.read_csv(path+'T2.csv')
print 'Shape T2 : ', T2.shape
Volume = pd.read_csv(path+'Volume.csv')
print 'Shape vol : ', Volume.shape
PatCode = pd.read_csv(path+'PatCode.csv')
print 'Shape PatCode : ', PatCode.shape
SessID = pd.read_csv(path+'SessID.csv')
print 'Shape SessID : ', SessID.shape
BirthAge = pd.read_csv(path+'BirthGA.csv')
print 'Shape BA : ', BirthAge.shape
ScanAge = pd.read_csv(path+'ScanGA.csv')
print 'Shape SA : ', ScanAge.shape
Gender = pd.read_csv(path+'Gender.csv')
print 'Shape Gender : ', Gender.shape
Term_labels = pd.read_csv(path+'Term_labels.csv')

T1 = T1[T1.columns[1:]]
T2 = T2[T2.columns[1:]]
Volume = Volume[Volume.columns[1:]]
SA = ScanAge[ScanAge.columns[1]]
BA = BirthAge[BirthAge.columns[1]]
PatCode = PatCode[PatCode.columns[1]]
SessID = SessID[SessID.columns[1]]
Gender = Gender[Gender.columns[1]]
Term_labels = Term_labels[Term_labels.columns[1]]

features = pd.concat([T1, T2, Volume], axis = 1)

training_data = []
testing_data = []
training_labels = []
testing_labels = []

best_features = []
raw_input('press ENTER')
training_scores = np.zeros((5,3))
training_accuracy = np.zeros((5,3))
train_std = np.zeros((5,3))
testing_scores = np.zeros((5,3))
testing_accuracy = np.zeros((5,3))
test_std = np.zeros((5,3))
feature_set = ()

rng = np.random.RandomState(42)

best_features = []
last_cv = 0;
feature_set = ()
best_set = ()
best_train_test = []


worst_cv = 99999
worst_set = ()
worst_train_test = []

N_est = [1, 10, 50, 100, 300]
ts = [0.5, 0.7, 0.9]
i = 0
j = 0
for n in N_est:
    for t in ts :
        training_data, testing_data, training_labels, testing_labels = train_test_split(features, Gender, train_size=t, random_state=rng)
        # print 'train data : ', len(training_data), 'train lab : ', len(training_labels)

        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, random_state=rng),
                                  n_estimators=n, random_state=rng)

        clf.fit(training_data, training_labels)

        y = (clf.predict(training_data))
        z = (clf.predict(testing_data))
        sy = (clf.score(training_data, training_labels))
        sz = (clf.score(testing_data, testing_labels))

        training_scores[i][j] = sy
        testing_scores[i][j] = sz

        # print ' '
        # print 'Training scores --> n_est = %0.2f , ts = %0.2f : ' % (n, t), sy
        # print 'Testing scores --> n_est = %0.2f , ts = %0.2f: '% (n, t), sz
        # print ' '

        cv_test = (cross_val_score(clf, features, Gender, cv=5))


        # print ' accuracy training: %0.2f (+/- %0.2f) ' % (cv_train.mean(), cv_train.std() *2)
        # print ' accuracy testing: %0.2f (+/- %0.2f) ' % (cv_test.mean(), cv_test.std() *2)
        testing_accuracy[i][j] = cv_test.mean()
        test_std[i][j] = cv_test.std()

        if cv_test.mean() > last_cv :
            best_features = clf.feature_importances_
            last_cv = cv_test.mean()
            feature_set = (n,t)
            best_set = (n,t)
            best_train_test = np.concatenate((training_labels, y, testing_labels, z))

        if cv_test.mean() < worst_cv :
            worst_cv = cv_test.mean()
            worst_set = (n,t)
            worst_train_test = np.concatenate((training_labels, y, testing_labels, z))
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # print 'basic score : ', clf.score(testing_data, testing_labels)
        j+=1
        if j > 2:
            j=0
    i+=1
    if i>4:
        i=0

print 'training scores final'
print training_scores
print 'testing scores final'
print testing_scores
print 'testing accuracy final'
print testing_accuracy
print 'testing std'
print test_std

zeros = np.zeros((5,1));
conc = np.concatenate((training_scores, zeros, testing_scores, zeros, testing_accuracy, zeros, test_std), axis = 1)
np.savetxt(path_save+'all_scores_gen_red_.csv', conc, fmt='%0.8f', delimiter=',')   # X is an array
np.savetxt(path_save+'best_features_gen_red_'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_features, fmt='%0.8f', delimiter=',')
np.savetxt(path_save+'best_scores_gen_red_'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_train_test, fmt='%s', delimiter=',')
np.savetxt(path_save+'worst_scores_gen_red_'+str(worst_set[0])+'_'+str(worst_set[1])+'.csv', worst_train_test, fmt='%s', delimiter=',')
# print 'conc'
# print conc
# print 'training scores final'
# print training_scores
# print 'testing scores final'
# print testing_scores
# print 'training accuracy final'
# print training_accuracy
# print 'testing accuracy final'
# print testing_accuracy
# print 'training std'
# print train_std
# print 'testing std'
# print test_std
