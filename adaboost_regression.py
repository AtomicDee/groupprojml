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
T2 = pd.read_csv(path+'T2.csv')
Volume = pd.read_csv(path+'Volume.csv')
BirthAge = pd.read_csv(path+'BirthGA.csv')


T1 = T1[T1.columns[1:]]
T2 = T2[T2.columns[1:]]
Vol = Volume[Volume.columns[1:]]
BA = BirthAge[BirthAge.columns[1]]
print BA

rnd = np.random.RandomState(42)
data = pd.concat([T1, T2, Vol], axis = 1)
labels = BA

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
best_train_test = []


worst_cv = 99999
worst_set = ()
worst_train_test = []


fig, ax = plt.subplots(5, 3)
fig, axes = plt.subplots(5, 3)
for n in N_est :
    for t in ts :

        training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, train_size=t, random_state=42)
        # Fit regression model
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4, random_state=rnd),
                                  n_estimators=n, random_state=rnd, learning_rate=0.1)
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
        diff = max(scores_z)-min(scores_z)
        print 'cv diff : ', diff
        print ' accuracy testing: %0.2f (+/- %0.2f) ' % (scores_z.mean(), scores_z.std() *2)


        testing_accuracy[i][j] = scores_z.mean()
        test_std[i][j] = scores_z.std()

        length_set = [len(training_labels)]

        if scores_z.mean() > last_cv :
            best_features = regr.feature_importances_
            last_cv = scores_z.mean()
            feature_set = (n,t)
            best_set = (n,t)
            best_train_test = np.concatenate((length_set, training_labels, y, testing_labels, z))

        if scores_z.mean() < worst_cv :
            worst_cv = scores_z.mean()
            worst_set = (n,t)
            worst_train_test = np.concatenate((length_set, training_labels, y, testing_labels, z))

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

plt.show()

print 'training scores final'
print training_scores
print 'testing scores final'
print testing_scores
print 'testing accuracy final'
print testing_accuracy
print 'testing std'
print test_std


# zeros = np.zeros((5,1));
# conc = np.concatenate((training_scores, zeros, testing_scores, zeros, testing_accuracy, zeros, test_std), axis = 1)
# np.savetxt('all_scores_full_.csv', conc, fmt='%0.8f', delimiter=',')   # X is an array
# np.savetxt('best_features_full_'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_features, fmt='%0.8f', delimiter=',')
# np.savetxt('best_scores_full_'+str(best_set[0])+'_'+str(best_set[1])+'.csv', best_train_test, fmt='%0.8f', delimiter=',')
# np.savetxt('worst_scores_full_'+str(worst_set[0])+'_'+str(worst_set[1])+'.csv', worst_train_test, fmt='%0.8f', delimiter=',')
print 'feature set'
print feature_set, '\n'
print 'Worst set'
print worst_set, '\n'


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
