# AdaBoost classification - use for feature extraction then svm
import pandas as pd
import nibabel
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile

from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# X, y = make_gaussian_quantiles(n_samples=1000, n_features=10,
#                                n_classes=3, random_state=1)
titles = ['Session ID','Birth Age','Scan Age','Gender','Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']
all_data = pd.read_csv('all_final_data.csv')

n_split = 300

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

print 'X_train : ', X_train.shape
print 'y_train : ', len(y_train)
