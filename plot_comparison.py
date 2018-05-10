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
# from scipy import pylab
import pylab
import scipy.optimize as opt
from scipy.optimize import curve_fit


# This is the function we are trying to fit to the data for the line line of best fit
def func(x, a, b):
     return a*x + b

# load in all of mickey's data
path_full = "/media/avi/MY USB 3/New folder/MICKEY/"

pred_BA_full_mick = pd.read_csv(path_full+'bestpred_SVM_BA_FullData.csv')
labels_BA_full_mick = (pd.read_csv(path_full+'bestpredLABELS_SVM_BA_FullData.csv'))
pred_SA_full_mick = pd.read_csv(path_full+'bestpred_SVM_SA_FullData.csv')
labels_SA_full_mick = (pd.read_csv(path_full+'bestpredLABELS_SVM_SA_FullData.csv'))
pred_BA_red_mick = pd.read_csv(path_full+'bestpred_SVM_BA_RData.csv')
labels_BA_red_mick = (pd.read_csv(path_full+'bestpredLABELS_SVM_BA_RData.csv'))
pred_SA_red_mick = pd.read_csv(path_full+'bestpred_SVM_SA_RData.csv')
labels_SA_red_mick = (pd.read_csv(path_full+'bestpredLABELS_SVM_SA_RData.csv'))

pred_BA_full_mick = pred_BA_full_mick[pred_BA_full_mick.columns[1:]]
labels_BA_full_mick = labels_BA_full_mick[labels_BA_full_mick.columns[1:]]
pred_SA_full_mick = pred_SA_full_mick[pred_SA_full_mick.columns[1:]]
labels_SA_full_mick = labels_SA_full_mick[labels_SA_full_mick.columns[1:]]
pred_BA_red_mick = pred_BA_red_mick[pred_BA_red_mick.columns[1:]]
labels_BA_red_mick = labels_BA_red_mick[labels_BA_red_mick.columns[1:]]
pred_SA_red_mick = pred_SA_red_mick[pred_SA_red_mick.columns[1:]]
labels_SA_red_mick = labels_SA_red_mick[labels_SA_red_mick.columns[1:]]
# print((pred_BA_full_mick[pred_BA_full_mick.columns[0]]))
# raw_input()
# print((labels_BA_full_mick))
# raw_input()

# load in all of avinash's paths - the data was generated each time on this plot for RF
path = '/media/avi/MY USB 3/New folder/GROUP_PROJECT/'
above_37_all_data_path = path + 'term+preterm/'
all_data_path = path + 'ALL_PATIENTS_NO_DUPLICATES/'

above_37_all_data_all_labels = pd.read_csv(above_37_all_data_path + 'Term_labels.csv')
above_37_all_data_all_t1t2vol = pd.read_csv(above_37_all_data_path + 'T1_T2_Vol.csv')
above_37_all_data_all_BirthGA = pd.read_csv(above_37_all_data_path + 'BirthGA.csv')
above_37_all_data_all_genders = pd.read_csv(above_37_all_data_path + 'Gender.csv')
above_37_all_data_all_ScanGA = pd.read_csv(above_37_all_data_path + 'ScanGA.csv')

all_data_all_labels = pd.read_csv(all_data_path + 'Term_labels.csv')
all_data_all_t1t2vol = pd.read_csv(all_data_path + 'T1_T2_Vol.csv')
all_data_all_BirthGA = pd.read_csv(all_data_path + 'BirthGA.csv')
all_data_all_genders = pd.read_csv(all_data_path + 'Gender.csv')
all_data_all_ScanGA = pd.read_csv(all_data_path + 'ScanGA.csv')

features_all = all_data_all_t1t2vol[all_data_all_t1t2vol.columns[1:]]
features_reduced = above_37_all_data_all_t1t2vol[above_37_all_data_all_t1t2vol.columns[1:]]
labels_reduced_BA = above_37_all_data_all_BirthGA[above_37_all_data_all_BirthGA.columns[1:]]
labels_all_BA = all_data_all_BirthGA[all_data_all_BirthGA.columns[2:]]
labels_reduced_SA = above_37_all_data_all_ScanGA[above_37_all_data_all_ScanGA.columns[1:]]
labels_all_SA = all_data_all_ScanGA[all_data_all_ScanGA.columns[2:]]


# load in all of daria's data
path_full = "/media/avi/MY USB 3/New folder/DARIA/"
path_red = "/media/avi/MY USB 3/New folder/DARIA/"


# Reduced Data
red_all_BA = pd.read_csv(path_red+'best_test_red_BA100_0.7.csv', header=None)
red_all_SA = pd.read_csv(path_red+'best_test_red_SA300_0.5.csv', header=None)

split_red_BA = len(red_all_BA)/2
split_red_SA = len(red_all_SA)/2

red_all_SA_lab = red_all_SA[:split_red_SA]
red_all_SA_pred = red_all_SA[split_red_SA:]
# print len(red_all_SA_lab), len(red_all_SA_pred)

red_all_BA_lab = red_all_BA[:split_red_BA]
red_all_BA_pred = red_all_BA[split_red_BA:]
# print len(red_all_BA_lab), len(red_all_BA_pred)

# Full Data
full_all_BA = pd.read_csv(path_full+'best_test_full_BA50_0.7.csv', header=None)
full_all_SA = pd.read_csv(path_full+'best_test_full_SA100_0.7.csv', header=None)

split_full_BA = len(full_all_BA)/2
split_full_SA = len(full_all_SA)/2

full_all_SA_lab = full_all_SA[:split_full_SA]
full_all_SA_pred = full_all_SA[split_full_SA:]
# print len(full_all_SA_lab), len(full_all_SA_pred)

full_all_BA_lab = full_all_BA[:split_full_BA]
full_all_BA_pred = full_all_BA[split_full_BA:]
# print len(full_all_BA_lab), len(full_all_BA_pred)

# Plotting
fig, ax = plt.subplots(2, 2)


# creating RF BA data on all data
depth=10
estim=300
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features_all, labels_all_BA, train_size=0.9, random_state=42)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_ba = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=42)
r_ba.fit(training_data,training_labels)
pred_ba = r_ba.predict(testing_data)
cv_score = cross_val_score(r_ba, testing_data, testing_labels)


#plotting all BA data
# plot svm data
ax[0,0].scatter(labels_BA_full_mick, pred_BA_full_mick, c='k', label='prediction', s=50.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, labels_BA_full_mick.squeeze(), pred_BA_full_mick.squeeze())
ax[0,0].plot(labels_BA_full_mick.squeeze(), func(labels_BA_full_mick.squeeze(), *optimizedParameters.squeeze()), label="fit", c='k')
# plot adaboost data
ax[0,0].scatter(full_all_BA_lab, full_all_BA_pred, c="c", label="Prediction", s=35.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, full_all_BA_lab.squeeze(), full_all_BA_pred.squeeze())
ax[0,0].plot(full_all_BA_lab.squeeze(), func(full_all_BA_lab.squeeze(), *optimizedParameters), label="fit", c='c')
# plot RF data
ax[0,0].scatter(testing_labels, pred_ba, c='y', label='Prediction', s=35.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, testing_labels.squeeze(), pred_ba.squeeze())
ax[0,0].plot(testing_labels.squeeze(), func(testing_labels.squeeze(), *optimizedParameters), label="fit", c='y')
ax[0,0].plot([0, 50],[0, 50],'--k', alpha=0.5)
ax[0,0].set_title("Full Set : Birth Age")
ax[0,0].set_xlabel('Actual')
ax[0,0].set_ylabel('Predicted')
ax[0,0].legend(['SVM', 'AdaBoost', 'RandomForest', 'Target'], loc=2)
ax[0,0].set_xlim([20, 45])
ax[0,0].set_ylim([20, 45])


# creating RF SA data on all data
depth=13
estim=50
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features_all, labels_all_SA, train_size=0.9, random_state=42)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_sa = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=42)
r_sa.fit(training_data,training_labels)
pred_sa = r_sa.predict(testing_data)
cv_score = cross_val_score(r_sa, testing_data, testing_labels)

# plotting all SA data
# plot svm data
ax[0,1].scatter(labels_SA_full_mick, pred_SA_full_mick, c='k', label='prediction', s=50.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, labels_SA_full_mick.squeeze(), pred_SA_full_mick.squeeze())
ax[0,1].plot(labels_SA_full_mick.squeeze(), func(labels_SA_full_mick.squeeze(), *optimizedParameters.squeeze()), label="fit", c='k')
# plot adaboost data
ax[0,1].scatter(full_all_SA_lab, full_all_SA_pred, c="c", label='Prediction', s=35)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, full_all_SA_lab.squeeze(), full_all_SA_pred.squeeze())
ax[0,1].plot(full_all_SA_lab.squeeze(), func(full_all_SA_lab.squeeze(), *optimizedParameters), label="fit", c='c')
# plot RF data
ax[0,1].scatter(testing_labels, pred_sa, c='y', label='Prediction', s=35.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, testing_labels.squeeze(), pred_ba.squeeze())
ax[0,1].plot(testing_labels.squeeze(), func(testing_labels.squeeze(), *optimizedParameters), label="fit", c='y')
ax[0,1].set_title("Full Set : Scan Age")
ax[0,1].plot([0, 50],[0, 50],'--k', alpha=0.5)
ax[0,1].set_xlabel('Actual')
ax[0,1].set_ylabel('Predicted')
ax[0,1].legend(['SVM', 'AdaBoost', 'RandomForest', 'Target'], loc=2)
ax[0,1].set_xlim([20, 45])
ax[0,1].set_ylim([20, 45])





# creating RF BA data on reduced data
depth=13
estim=50
split=2
training_data, testing_data, training_labels, testing_labels = train_test_split(features_reduced, labels_reduced_BA, train_size=0.9, random_state=42)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_ba = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=42)
r_ba.fit(training_data,training_labels)
pred_ba = r_ba.predict(testing_data)
cv_score = cross_val_score(r_ba, testing_data, testing_labels)

# plotting reduced BA data
# plot svm data
ax[1,0].scatter(labels_BA_red_mick, pred_BA_red_mick, c='k', label='prediction', s=50.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, labels_BA_red_mick.squeeze(), pred_BA_red_mick.squeeze())
ax[1,0].plot(labels_BA_red_mick.squeeze(), func(labels_BA_red_mick.squeeze(), *optimizedParameters.squeeze()), label="fit", c='k')
# plot adaboost data
ax[1,0].scatter(red_all_BA_lab, red_all_BA_pred, c="c", label='Prediction', s=35)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, red_all_BA_lab.squeeze(), red_all_BA_pred.squeeze())
ax[1,0].plot(red_all_BA_lab.squeeze(), func(red_all_BA_lab.squeeze(), *optimizedParameters), label="fit", c='c')
# plot RF data
ax[1,0].scatter(testing_labels, pred_ba, c='y', label='Prediction', s=35.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, testing_labels.squeeze(), pred_ba.squeeze())
ax[1,0].plot(testing_labels.squeeze(), func(testing_labels.squeeze(), *optimizedParameters), label="fit", c='y')
ax[1,0].set_title("Reduced Set : Birth Age" )
ax[1,0].plot([0, 50],[0, 50],'--k', alpha=0.5)
ax[1,0].set_xlabel('Actual')
ax[1,0].set_ylabel('Predicted')
ax[1,0].legend(['SVM', 'AdaBoost', 'RandomForest', 'Target'], loc=2)
ax[1,0].set_xlim([20, 45])
ax[1,0].set_ylim([20, 45])

# creating RF SA data on reduced data
depth=7
estim=100
split=4
training_data, testing_data, training_labels, testing_labels = train_test_split(features_reduced, labels_reduced_SA, train_size=0.7, random_state=42)
testing_labels = np.ravel(testing_labels)
training_labels = np.ravel(training_labels)
r_sa = RandomForestRegressor(max_depth=depth, max_features=16, n_estimators=estim, min_samples_split=split, random_state=42)
r_sa.fit(training_data,training_labels)
pred_sa = r_sa.predict(testing_data)
cv_score = cross_val_score(r_sa, testing_data, testing_labels)

# plotting reduced SA data
# plot svm data
ax[1,1].scatter(labels_SA_red_mick, pred_SA_red_mick, c='k', label='prediction', s=50.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, labels_SA_red_mick.squeeze(), pred_SA_red_mick.squeeze())
ax[1,1].plot(labels_SA_red_mick.squeeze(), func(labels_SA_red_mick.squeeze(), *optimizedParameters.squeeze()), label="fit", c='k')
# plot adaboost data
ax[1,1].scatter(red_all_SA_lab, red_all_SA_pred, c="c", label='Prediction', s=35)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, red_all_SA_lab.squeeze(), red_all_SA_pred.squeeze())
ax[1,1].plot(red_all_SA_lab.squeeze(), func(red_all_SA_lab.squeeze(), *optimizedParameters), label="fit", c='c')
# plot RF data
ax[1,1].scatter(testing_labels, pred_sa, c='y', label='Prediction', s=35.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, testing_labels.squeeze(), pred_sa.squeeze())
ax[1,1].plot(testing_labels.squeeze(), func(testing_labels.squeeze(), *optimizedParameters), label="fit", c='y')
ax[1,1].set_title("Reduced Set : Scan Age" )
ax[1,1].plot([0, 50],[0, 50],'--k', alpha=0.5)
ax[1,1].set_xlabel('Actual')
ax[1,1].set_ylabel('Predicted')
ax[1,1].legend(['SVM', 'AdaBoost', 'RandomForest', 'Target'], loc=2)
ax[1,1].set_xlim([20, 45])
ax[1,1].set_ylim([20, 45])


# plot the reduced SA separately on axis to a better scale
plt.figure(2)
# plot svm data
plt.scatter(labels_SA_red_mick, pred_SA_red_mick, c='k', label='prediction', s=50.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, labels_SA_red_mick.squeeze(), pred_SA_red_mick.squeeze())
plt.plot(labels_SA_red_mick.squeeze(), func(labels_SA_red_mick.squeeze(), *optimizedParameters.squeeze()), label="fit", c='k')
# plot adaboost data
plt.scatter(red_all_SA_lab, red_all_SA_pred, c="c", label='Prediction', s=35)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, red_all_SA_lab.squeeze(), red_all_SA_pred.squeeze())
plt.plot(red_all_SA_lab.squeeze(), func(red_all_SA_lab.squeeze(), *optimizedParameters), label="fit", c='c')
# plot RF data
plt.scatter(testing_labels, pred_sa, c='y', label='Prediction', s=35.0)
# generate line of best fit
optimizedParameters, pcov = opt.curve_fit(func, testing_labels.squeeze(), pred_sa.squeeze())
plt.plot(testing_labels.squeeze(), func(testing_labels.squeeze(), *optimizedParameters), label="fit", c='y')
plt.title("Reduced Set : Scan Age" )
plt.plot([0, 50],[0, 50],'--k', alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend(['SVM', 'AdaBoost', 'RandomForest', 'Target'], loc=2)
plt.xlim([35, 45])
plt.ylim([35, 45])






plt.show()
