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
print T1
raw_input('press ENTER to continue')
T2 = pd.read_csv(path+'T2.csv', sep=';')
Volume = pd.read_csv(path+'Volume.csv', sep=';')
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

training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, train_size=0.8)

# Fit regression model
regr_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), random_state=rng)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=150, random_state=rng)

regr_3 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(training_data, training_labels)
regr_2.fit(training_data, training_labels)
regr_3.fit(training_data, training_labels)




# Predict
y_1 = regr_1.predict(training_data)
print y_1.shape
y_2 = regr_2.predict(training_data)
y_3 = regr_3.predict(training_data)

z_1 = regr_1.predict(testing_data)
z_2 = regr_2.predict(testing_data)
z_3 = regr_3.predict(testing_data)


# sy_1 = regr_1.score(training_data, training_labels)
# sy_2 = regr_2.score(training_data, training_labels)
# sy_3 = regr_3.score(training_data, training_labels)
#
# sz_1 = regr_1.score(testing_data, testing_labels)
# sz_2 = regr_2.score(testing_data, testing_labels)
# sz_3 = regr_3.score(testing_data, testing_labels)
#
# print ' '
# print 'Training scores --> n_est = 1 : ', sy_1, ' n_est = 300 : ', sy_2, ' n_est = 1000 : ', sy_3
# print 'Testing scores --> n_est = 1 : ', sz_1, ' n_est = 300 : ', sz_2, ' n_est = 1000 : ', sz_3
# print ' '
# # data to list
#
# scores_z1 = cross_val_score(regr_1, testing_data, testing_labels)
# scores_z2 = cross_val_score(regr_2, testing_data, testing_labels)
# scores_z3 = cross_val_score(regr_3, testing_data, testing_labels)
#
# scores_y1 = cross_val_score(regr_1, training_data, training_labels)
# scores_y2 = cross_val_score(regr_2, training_data, training_labels)
# scores_y3 = cross_val_score(regr_3, training_data, training_labels)
# print ' accuracy z1: %0.2f (+/- %0.2f) ' % (scores_z1.mean(), scores_z1.std() *2)
# print ' accuracy z2: %0.2f (+/- %0.2f) ' % (scores_z2.mean(), scores_z2.std() *2)
# print ' accuracy z3: %0.2f (+/- %0.2f) ' % (scores_z3.mean(), scores_z3.std() *2)
# print ' accuracy y1: %0.2f (+/- %0.2f) ' % (scores_y1.mean(), scores_y1.std() *2)
# print ' accuracy y2: %0.2f (+/- %0.2f) ' % (scores_y2.mean(), scores_y2.std() *2)
# print ' accuracy y3: %0.2f (+/- %0.2f) ' % (scores_y3.mean(), scores_y3.std() *2)
#
# # list_labels = np.concatenate(training_labels.values).ravel().tolist()
#
# # Plot the results
# plt.figure()
# plt.scatter(training_labels, y_1, c="g", label="n_estimators=1", linewidth=2)
# plt.scatter(training_labels, y_2, c="r", label="n_estimators=300", linewidth=2)
# plt.scatter(training_labels, y_3, c="b", label="n_estimators=1000", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("Predictions")
# plt.title("training data")
# plt.legend()
#
# plt.figure()
# plt.scatter(testing_labels, z_1, c="g", label="n_estimators=1", linewidth=2)
# plt.scatter(testing_labels, z_2, c="r", label="n_estimators=300", linewidth=2)
# plt.scatter(testing_labels, z_3, c="b", label="n_estimators=1000", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("Predictions")
# plt.title("Testing data")
# plt.legend()
#
# plt.show()

# # feature extraction
fi_3 = regr_3.feature_importances_
print len(fi_3)

T1_feature_scores = fi_3[:86]
print len(T1_feature_scores)
print type(T1_feature_scores)
T2_feature_scores = fi_3[86:172]
print len(T2_feature_scores)
Vol_feature_scores = fi_3[172:]
print len(Vol_feature_scores)

df1 = pd.DataFrame(T1_feature_scores)
df1.to_csv(path+"new_T1_300.csv", header=None, index=None)

df2 = pd.DataFrame(T2_feature_scores)
df2.to_csv(path+"new_T2_300.csv", header=None, index=None)

df3 = pd.DataFrame(Vol_feature_scores)
df3.to_csv(path+"new_Volume_300.csv", header=None, index=None)
