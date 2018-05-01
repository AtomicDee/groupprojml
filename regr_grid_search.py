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
from sklearn.model_selection import GridSearchCV

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
X = pd.concat([T1, T2, Vol], axis = 1)
y = BA

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=rnd)

# Adaboost params : base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=None

# Decision params : criterion='mse', max_depth=None, max_features=None,
# max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, min_samples_leaf=1,
# min_samples_split=2, min_weight_fraction_leaf=0.0,
# presort=False, random_state=None, splitter='best'

# Set the parameters by cross-validation
tuned_parameters = [{'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 1],
                        'loss': ['linear','square','exponential'],
                        'n_estimators': [1,10,50,100,300,500,1000]}]


scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    dtr = DecisionTreeRegressor(random_state=42, min_samples_split= 6, criterion='friedman_mse', max_depth=6)
    abr = AdaBoostRegressor(base_estimator=dtr, random_state=42)
    regr = GridSearchCV(abr, tuned_parameters, cv=5)
    regr.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(regr.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = regr.cv_results_['mean_test_score']
    stds = regr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, regr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, regr.predict(X_test)
    # print(regression_report(y_true, y_pred))
    # print()
