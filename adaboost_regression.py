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

path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
T1 = pd.read_csv(path+'T1.csv')
#print T1
#raw_input('press ENTER to continue')
T2 = pd.read_csv(path+'T2.csv')
Volume = pd.read_csv(path+'Volume.csv')
BirthAge = pd.read_csv(path+'BirthGA.csv')
ScanAge = pd.read_csv(path+'ScanGA.csv')

T1 = T1[T1.columns[1:]]
T2 = T2[T2.columns[1:]]
Vol = Volume[Volume.columns[1:]]
SA = ScanAge[ScanAge.columns[1]]
BA = BirthAge[BirthAge.columns[1]]

rng = np.random.RandomState(1)
data = pd.concat([T1, T2, Vol], axis = 1)
print data.shape
labels = SA

N_est = [1, 10, 50, 100, 300]
ts = [0.2, 0.5, 0.8]
i = 0
j = 0
fig, ax = plt.subplots(5, 3)
fig, axes = plt.subplots(5, 3)
for n in N_est :
    for t in ts :

        training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, train_size=t)

        # Fit regression model
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                  n_estimators=150, random_state=rng)
        regr.fit(training_data, training_labels)

        # Predict
        y = regr.predict(training_data)
        z = regr.predict(testing_data)
        sy = regr.score(training_data, training_labels)
        sz = regr.score(testing_data, testing_labels)

        print ' '
        print 'Training scores --> n_est = %0.2f , ts = %0.2f : ' % (n, t), sy
        print 'Testing scores --> n_est = %0.2f , ts = %0.2f: '% (n, t), sz
        print ' '

        scores_z = cross_val_score(regr, testing_data, testing_labels)
        scores_y = cross_val_score(regr, training_data, training_labels)
        print ' accuracy training: %0.2f (+/- %0.2f) ' % (scores_y.mean(), scores_y.std() *2)
        print ' accuracy testing: %0.2f (+/- %0.2f) ' % (scores_z.mean(), scores_z.std() *2)

        # list_labels = np.concatenate(training_labels.values).ravel().tolist()

        # Plot the results

        print i, j
        axes[i,j].scatter(training_labels, y, c="g", label="n_estimators=%0.2f" % n, linewidth=2)
        axes[i,j].set_title("train n :%0.2f ts:%0.2f" % (n, t))

        ax[i,j].scatter(testing_labels, z, c="g", label="n_estimators=%0.2f" % n, linewidth=2)
        ax[i,j].set_title("test n :%0.2f ts:%0.2f" % (n, t))
        j+=1
        if j > 2:
            j=0
    i+=1
    if i>4:
        i=0

plt.show()

# # feature extraction
# fi_3 = regr_3.feature_importances_
# print len(fi_3)
#
# T1_feature_scores = fi_3[:86]
# print len(T1_feature_scores)
# print type(T1_feature_scores)
# T2_feature_scores = fi_3[86:172]
# print len(T2_feature_scores)
# Vol_feature_scores = fi_3[172:]
# print len(Vol_feature_scores)
#
# df1 = pd.DataFrame(T1_feature_scores)
# df1.to_csv(path+"new_T1_300.csv", header=None, index=None)
#
# df2 = pd.DataFrame(T2_feature_scores)
# df2.to_csv(path+"new_T2_300.csv", header=None, index=None)
#
# df3 = pd.DataFrame(Vol_feature_scores)
# df3.to_csv(path+"new_Volume_300.csv", header=None, index=None)
