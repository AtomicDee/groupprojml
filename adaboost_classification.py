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
path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"

T1 = pd.read_csv(path+'T1')
T2 = pd.read_csv(path+'T2')
Volume = pd.read_csv(path+'Volume')
PatCode = pd.read_csv(path+'PatCode')
SessID = pd.read_csv(path+'SessID')
BirthAge = pd.read_csv(path+'BirthAge')
ScanAge = pd.read_csv(path+'ScanAge')
Gender = pd.read_csv(path+'Gender')

train_t1 = []
test_t1 = []
i = 1

for row in T1.itertuples():
    if i % 2 == 0:
        train_t1.append(pd.DataFrame([row]))
    else:
        test_t1.append(pd.DataFrame([row]))
        i += 1
train_t1 = pd.concat(train_t1)
test_t1 = pd.concat(test_t1)


training_data = train_t1
training_labels = Gender
