#Adaboost regression
import pandas as pd
import nibabel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Max features = some sample from 87*3
# max max_depth
# n estimators

# path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/split_data/term_preterm/"
path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/ALL_PATIENTS_NO_DUPLICATES/"
path_save = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/reg_full_results/"
T1 = pd.read_csv(path+'T1.csv')
T2 = pd.read_csv(path+'T2.csv')
Volume = pd.read_csv(path+'Volume.csv')
BirthAge = pd.read_csv(path+'BirthGA.csv')
ScanAge = pd.read_csv(path+'ScanGA.csv')

T1 = T1[T1.columns[2:]]
T2 = T2[T2.columns[2:]]
Vol = Volume[Volume.columns[2:]]
BA = BirthAge[BirthAge.columns[2]]
SA = ScanAge[ScanAge.columns[2]]
print SA

rnd = np.random.RandomState(42)
data = pd.concat([T1, T2, Vol], axis = 1)
labels = BA

# Optimised Parameters for DecisionTreeRegressor :
# random_state=42, min_samples_split= 6, criterion='friedman_mse', max_depth=6

# Optimised Parameters for AdaBoostRegressor with above DecisionTreeRegressor parameters :



N_est = [1, 10, 50, 100, 300]
ts = [0.5, 0.7, 0.9]
i = 0
j = 0

training_scores = np.zeros((5,3))
testing_scores = np.zeros((5,3))
testing_accuracy = np.zeros((5,3))
test_std = np.zeros((5,3))


best_features = []
last_cv = 0;
feature_set = ()
best_set = ()
best_train = []
best_test = []


worst_cv = 99999
worst_set = ()
worst_train = []
worst_test = []


fig, ax = plt.subplots(5, 3)
fig, axes = plt.subplots(5, 3)
for n in N_est :
    for t in ts :

        training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, train_size=t, random_state=42)
        # Fit regression model
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2, random_state=rnd),
                                  n_estimators=n, random_state=rnd, learning_rate=1)
        regr.fit(training_data, training_labels)

        # Predict
        y = regr.predict(training_data)
        z = regr.predict(testing_data)
        sy = regr.score(training_data, training_labels)
        sz = regr.score(testing_data, testing_labels)

        training_scores[i][j] = sy
        testing_scores[i][j] = sz

        print ' '
        print 'Training scores --> n_est = %0.2f , ts = %0.2f : ' % (n, t), sy
        print 'Testing scores --> n_est = %0.2f , ts = %0.2f: '% (n, t), sz
        print ' '

        scores_z = cross_val_score(regr, data, labels, cv=5)
        scores_test = cross_val_score(regr, testing_data, testing_labels, cv=5)
        diff = max(scores_z)-min(scores_z)
        print 'cv diff : ', diff
        print ' accuracy testing: %0.2f (+/- %0.2f) ' % (scores_test.mean(), scores_test.std() *2)
        print ' accuracy all: %0.2f (+/- %0.2f) ' % (scores_z.mean(), scores_z.std() *2)


        testing_accuracy[i][j] = scores_test.mean()
        test_std[i][j] = scores_test.std()

        length_set = [len(training_labels)]

        if scores_test.mean() > last_cv :
            best_features = regr.feature_importances_
            last_cv = scores_test.mean()
            feature_set = (n,t)
            best_set = (n,t)
            best_train = np.concatenate((training_labels, y))
            best_test = np.concatenate((testing_labels, z))

        if scores_test.mean() < worst_cv :
            worst_cv = scores_test.mean()
            worst_set = (n,t)
            worst_train = np.concatenate((training_labels, y))
            worst_test = np.concatenate((testing_labels, z))

        # list_labels = np.concatenate(training_labels.values).ravel().tolist()

        # Plot the results

        print i, j
        axes[i,j].scatter(training_labels, y, c="g", label="n_estimators=%0.2f" % n, linewidth=1)
        axes[i,j].set_title("train n :%0.2f ts:%0.2f" % (n, t))


        ax[i,j].scatter(testing_labels, z, c="g", label="n_estimators=%0.2f" % n, linewidth=1)
        ax[i,j].set_title("test n :%0.2f ts:%0.2f" % (n, t))

        j+=1
        if j > 2:
            j=0
    i+=1
    if i>4:
        i=0

#plt.show()

# print 'training scores final'
# print training_scores
# print 'testing scores final'
# print testing_scores
# print 'testing accuracy final'
# print testing_accuracy
# print 'testing std'
# print test_std


# zeros = np.zeros((5,1));
# conc = np.concatenate((training_scores, zeros, testing_scores, zeros, testing_accuracy, zeros, test_std), axis = 1)
# np.savetxt(path_save+'all_scores_red_BA.csv', conc, fmt='%0.8f', delimiter=',')   # X is an array
# np.savetxt(path_save+'best_features_red_BA'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_features, fmt='%0.8f', delimiter=',')
np.savetxt(path_save+'best_train_full_BA'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_train, fmt='%0.8f', delimiter=',')
np.savetxt(path_save+'best_test_full_BA'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_test, fmt='%0.8f', delimiter=',')
np.savetxt(path_save+'worst_train_full_BA'+str(worst_set[0])+'_'+str(worst_set[1])+'.csv', worst_train, fmt='%0.8f', delimiter=',')
np.savetxt(path_save+'worst_test_full_BA'+str(worst_set[0])+'_'+str(worst_set[1])+'.csv', worst_test, fmt='%0.8f', delimiter=',')
print 'feature set'
print feature_set, '\n'
print 'Worst set'
print worst_set, '\n'


# feature extraction

# T1_feature_scores = best_features[:86]
# print len(T1_feature_scores)
# print type(T1_feature_scores)
# T2_feature_scores = best_features[86:172]
# print len(T2_feature_scores)
# Vol_feature_scores = best_features[172:]
# print len(Vol_feature_scores)
#
# df1 = pd.DataFrame(T1_feature_scores)
# df1.to_csv(path_save+"new_T1_"+str(best_set[0])+'_'+str(best_set[1])+'.csv', header=None, index=None)
#
# df2 = pd.DataFrame(T2_feature_scores)
# df2.to_csv(path_save+"new_T2_"+str(best_set[0])+'_'+str(best_set[1])+'.csv', header=None, index=None)
#
# df3 = pd.DataFrame(Vol_feature_scores)
# df3.to_csv(path_save+"new_Volume_"+str(best_set[0])+'_'+str(best_set[1])+'.csv', header=None, index=None)
