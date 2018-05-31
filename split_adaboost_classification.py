# AdaBoost classification - use for feature extraction then svm
# Us specifically with split data files
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
#Load Term Infant Data
path_term = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/split_data/Term_Data/"
Term_data = pd.read_csv(path_term+'T1_T2_Vol.csv')
BirthAge_term = pd.read_csv(path_term+'BirthGA.csv')
ScanAge_term = pd.read_csv(path_term+'ScanGA.csv')
l_term = pd.read_csv(path_term+'Term_labels.csv')

t_data = Term_data[Term_data.columns[1:]]
SA_t = ScanAge_term[ScanAge_term.columns[1]]
BA_t = BirthAge_term[BirthAge_term.columns[1]]
l_t = l_term[l_term.columns[1]]


#Load preterm infant data
path_preterm = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/split_data/preterm_abov_37/"
Preterm_data = pd.read_csv(path_preterm+'T1_T2_Vol.csv')
BirthAge_pre = pd.read_csv(path_preterm+'BirthGA.csv')
ScanAge_pre = pd.read_csv(path_preterm+'ScanGA.csv')
l_preterm = pd.read_csv(path_preterm+'Term_labels.csv')

p_data = Preterm_data[Preterm_data.columns[1:]]
SA_p = ScanAge_pre[ScanAge_pre.columns[1]]
BA_p = BirthAge_pre[BirthAge_pre.columns[1]]
l_p = l_preterm[l_preterm.columns[1]]

rng = np.random.RandomState(1)
training_data = pd.concat([t_data,SA_t], axis = 1)
training_labels = l_t

#raw_input('ENTER')

testing_data = pd.concat([p_data,SA_p], axis=1)
testing_labels = l_p

best_features = []

#raw_input('press ENTER')

#Initialise result matrices
training_scores = np.zeros((5,3))
training_accuracy = np.zeros((5,3))
train_std = np.zeros((5,3))
testing_scores = np.zeros((5,3))
testing_accuracy = np.zeros((5,3))
test_std = np.zeros((5,3))
feature_set = ()

rng = np.random.RandomState(42)

last_cv = 0;

# interate number of estimates
N_est = [1, 10, 50, 100, 300]

i = 0
j = 0
for n in N_est:

    # print 'train data : ', len(training_data), 'train lab : ', len(training_labels)

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                              n_estimators=n, random_state=rng)

    clf.fit(training_data, training_labels)

    y = (clf.predict(training_data))
    z = (clf.predict(testing_data))
    sy = (clf.score(training_data, training_labels))
    sz = (clf.score(testing_data, testing_labels))

    #Regular accuracy scoring
    training_scores[i][j] = sy
    testing_scores[i][j] = sz

    print ' '
    print 'Training scores --> n_est = %0.2f  : ' % (n), sy
    print 'Testing scores --> n_est = %0.2f : '% (n), sz
    print ' '
    #Cross Validaiton scoring
    cv_test = (cross_val_score(clf, testing_data, testing_labels))
    cv_train = (cross_val_score(clf, training_data, training_labels))

    print ' accuracy training: %0.2f (+/- %0.2f) ' % (cv_train.mean(), cv_train.std() *2)
    print ' accuracy testing: %0.2f (+/- %0.2f) ' % (cv_test.mean(), cv_test.std() *2)
    training_accuracy[i][j] = cv_train.mean()
    testing_accuracy[i][j] = cv_test.mean()
    train_std[i][j] = cv_train.std()
    test_std[i][j] = cv_test.std()

    #Save separate set of best features based on highest cross val scores
    if cv_test.mean() > last_cv :
        best_features = clf.feature_importances_
        last_cv = cv_test.mean()
        feature_set = (n)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # print 'basic score : ', clf.score(testing_data, testing_labels)
    j+=1
    if j > 4:
        j=0

# Save results, added zeros column in between tables for easy splitting without knowing the dimensions of the results tables.
zeros = np.zeros((5,1));
conc = np.concatenate((training_scores, zeros, testing_scores, zeros, training_accuracy, zeros, testing_accuracy, zeros, train_std, zeros, test_std), axis = 1)
np.savetxt('test.csv', conc, fmt='%0.8f', delimiter=',')   # X is an array
np.savetxt('features.csv', best_features, fmt='%0.8f', delimiter=',')
print 'feature set'
print feature_set, '\n'
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
