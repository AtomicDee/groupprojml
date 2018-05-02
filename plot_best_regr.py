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
# path = '/home/avi/Desktop/groupprojml/DATA/All_Patients'
# path = '/home/avi/Desktop/Final_Data_groupprojml/' #for all data
# path = 'C:\\Users\\Avi\\Desktop\\GROUP PROJECT\\'
path = 'C:/Users/Avi/Desktop/New folder/GROUP_PROJECT/'
path = '/media/avi/MY USB 3/New folder/GROUP_PROJECT/'
above_37_all_data_path = path + 'term+preterm/'
preterm_path = path + 'preterm_above_37/'
all_data_path = path + 'unsplit_data/'
term_path = path + 'Term_Data/'

above_37_all_data_all_features = pd.read_csv(above_37_all_data_path + 'ALL_FEATURES.csv')
above_37_all_data_all_labels = pd.read_csv(above_37_all_data_path + 'Term_labels.csv')
above_37_all_data_all_t1t2vol = pd.read_csv(above_37_all_data_path + 'T1_T2_Vol.csv')
above_37_all_data_all_BirthGA = pd.read_csv(above_37_all_data_path + 'BirthGA.csv')
above_37_all_data_all_genders = pd.read_csv(above_37_all_data_path + 'Gender.csv')
above_37_all_data_all_ScanGA = pd.read_csv(above_37_all_data_path + 'ScanGA.csv')

all_data_all_features = pd.read_csv(all_data_path + 'ALL_FEATURES.csv')
all_data_all_labels = pd.read_csv(all_data_path + 'Term_labels.csv')
all_data_all_t1t2vol = pd.read_csv(all_data_path + 'T1_T2_Vol.csv')
all_data_all_BirthGA = pd.read_csv(all_data_path + 'BirthGA.csv')
all_data_all_genders = pd.read_csv(all_data_path + 'Gender.csv')
all_data_all_ScanGA = pd.read_csv(all_data_path + 'ScanGA.csv')

preterm_all_features = pd.read_csv(preterm_path + 'ALL_FEATURES.csv')
preterm_all_labels = pd.read_csv(preterm_path + 'Term_labels.csv')
preterm_all_t1t2vol = pd.read_csv(preterm_path + 'T1_T2_Vol.csv')
preterm_all_BirthGA = pd.read_csv(preterm_path + 'BirthGA.csv')
preterm_all_genders = pd.read_csv(preterm_path + 'Gender.csv')
preterm_all_ScanGA = pd.read_csv(preterm_path + 'ScanGA.csv')

term_all_features = pd.read_csv(term_path + 'ALL_FEATURES.csv')
term_all_labels = pd.read_csv(term_path + 'Term_labels.csv')
term_all_t1t2vol = pd.read_csv(term_path + 'T1_T2_Vol.csv')
term_all_BirthGA = pd.read_csv(term_path + 'BirthGA.csv')
term_all_genders = pd.read_csv(term_path + 'Gender.csv')
term_all_ScanGA = pd.read_csv(term_path + 'ScanGA.csv')

# full data
# features = all_data_all_t1t2vol[all_data_all_t1t2vol.columns[1:]]
# features = all_data_all_features[all_data_all_features.columns[1:]]
# labels = all_data_all_genders[all_data_all_genders.columns[1:]]
# labels = all_data_all_BirthGA[all_data_all_BirthGA.columns[1:]]
model = 'clf'
# features = above_37_all_data_all_t1t2vol[above_37_all_data_all_t1t2vol.columns[1:]]
features = above_37_all_data_all_features[above_37_all_data_all_features.columns[1:]]
# labels = above_37_all_data_all_genders[above_37_all_data_all_genders.columns[1:]]
labels = above_37_all_data_all_BirthGA[above_37_all_data_all_BirthGA.columns[1:]]

RS1=153
RS2=89
depth=10
estim=100
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=0.5, random_state=RS1)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_05_reduced = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=RS2)
r_05_reduced.fit(training_data,training_labels)
pred_05_reduced = r_05_reduced.predict(testing_data)
cv_score = cross_val_score(r_05_reduced, testing_data, testing_labels)

plt.subplot(2, 3, 1)
plt.scatter(testing_labels, pred_05_reduced, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim([20, 50])
plt.ylim([20, 50])
plt.title('0.5 - reduced - cv:'+str(np.mean(cv_score)))
plt.legend()

RS1=135
RS2=14
depth=7
estim=300
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=0.7, random_state=RS1)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_07_reduced = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=RS2)
r_07_reduced.fit(training_data,training_labels)
pred_07_reduced = r_07_reduced.predict(testing_data)
cv_score = cross_val_score(r_07_reduced, testing_data, testing_labels)

plt.subplot(2, 3, 2)
plt.scatter(testing_labels, pred_07_reduced, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim([20, 50])
plt.ylim([20, 50])
plt.title('0.7 - reduced - cv:'+str(np.mean(cv_score)))
plt.legend()

RS1=27
RS2=128
depth=10
estim=50
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=0.9, random_state=RS1)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_09_reduced = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=RS2)
r_09_reduced.fit(training_data,training_labels)
pred_09_reduced = r_09_reduced.predict(testing_data)
cv_score = cross_val_score(r_09_reduced, testing_data, testing_labels)

plt.subplot(2, 3, 3)
plt.scatter(testing_labels, pred_09_reduced, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim([20, 50])
plt.ylim([20, 50])
plt.title('0.9 - reduced - cv:'+str(np.mean(cv_score)))
plt.legend()


# features = all_data_all_t1t2vol[all_data_all_t1t2vol.columns[1:]]
features = all_data_all_features[all_data_all_features.columns[1:]]
# labels = all_data_all_genders[all_data_all_genders.columns[1:]]
labels = all_data_all_BirthGA[all_data_all_BirthGA.columns[1:]]


RS1=146
RS2=132
depth=50
estim=100
split=4
training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=0.5, random_state=RS1)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_05_reduced = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=RS2)
r_05_reduced.fit(training_data,training_labels)
pred_05_reduced = r_05_reduced.predict(testing_data)
cv_score = cross_val_score(r_05_reduced, testing_data, testing_labels)

plt.subplot(2, 3, 4)
plt.scatter(testing_labels, pred_05_reduced, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim([20, 50])
plt.ylim([20, 50])
plt.title('0.5 - full - cv:'+str(np.mean(cv_score)))
plt.legend()

RS1=106
RS2=176
depth=10
estim=300
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=0.7, random_state=RS1)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_07_reduced = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=RS2)
r_07_reduced.fit(training_data,training_labels)
pred_07_reduced = r_07_reduced.predict(testing_data)
cv_score = cross_val_score(r_07_reduced, testing_data, testing_labels)

plt.subplot(2, 3, 5)
plt.scatter(testing_labels, pred_07_reduced, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim([20, 50])
plt.ylim([20, 50])
plt.title('0.7 - full - cv:'+str(np.mean(cv_score)))
plt.legend()

RS1=3
RS2=47
depth=10
estim=50
split=4
training_data, testing_data, training_labels, testing_labels = train_test_split(features, labels, train_size=0.9, random_state=RS1)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_09_reduced = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=RS2)
r_09_reduced.fit(training_data,training_labels)
pred_09_reduced = r_09_reduced.predict(testing_data)
cv_score = cross_val_score(r_09_reduced, testing_data, testing_labels)

plt.subplot(2, 3, 6)
plt.scatter(testing_labels, pred_09_reduced, c='b', label='Prediction')
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.xlim([20, 50])
plt.ylim([20, 50])
plt.title('0.9 - full - cv:'+str(np.mean(cv_score)))
plt.legend()

plt.show()
