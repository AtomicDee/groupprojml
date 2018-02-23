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

path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
T1 = pd.read_csv(path+'T1.csv')
T2 = pd.read_csv(path+'T2.csv')
Volume = pd.read_csv(path+'Volume.csv')
BirthAge = pd.read_csv(path+'BirthAge.csv')
ScanAge = pd.read_csv(path+'ScanAge.csv')

T1 = T1[T1.columns[2:]]
T2 = T2[T2.columns[2:]]
Vol = Volume[Volume.columns[2:]]
SA = ScanAge[ScanAge.columns[1]]
BA = BirthAge[BirthAge.columns[1]]

rng = np.random.RandomState(1)
data = pd.concat([T1, T2, Vol], axis = 1)
labels = SA

training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, train_size=0.5)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(training_data, training_labels)
regr_2.fit(training_data, training_labels)

# Predict
y_1 = regr_1.predict(training_data)
y_2 = regr_2.predict(training_data)


# data to list

list_data = np.concatenate(training_data.values).ravel().tolist()

# Plot the results
plt.figure()
plt.scatter(list_data, training_labels, c="k", label="training samples")
plt.plot(list_data, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(list_data, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
