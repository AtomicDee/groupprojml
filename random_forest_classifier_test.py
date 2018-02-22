import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
# from sklearn import export_graphviz
import nibabel

# long as code thing, numbering, region numbers, T1, T2, Volume (test1.csv)

fields = ['Session ID', 'Birth Age', 'Scan Age', 'Gender', 'Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']
# Select which columns to load

# path = '/home/avi/Documents/groupprojml/test1.csv'

# The path where all the data is held
path = "/Users/daria/Documents/Group diss/Group Project Data/Data/Reformatted_data.csv"
data = pd.read_csv(path, skipinitialspace=True, usecols=fields)
# Using pandas to read all the data in

sessID.append = data[row[3]]
GAB = data['Birth Age']
GAS = data['Scan Age']
gender = data['Gender']
region = data['Region']
T1 = data['T1 Average Intensity']
T2 = data['T2 Average Intensity']
Vol = data['Volume']
# Assigning variables

training_data = T2.values.reshape(-1, 1)
training_labels = gender

clf = RandomForestClassifier()
# clf = RandomForestClassifier(criterion = "mse", splitter = "best", max_depth = 3,
#                             min_samples_split = 2, max_features = 2)
clf = clf.fit(training_data, training_labels)

tree.export_graphviz(clf)

# clf.predict(test_data, training_labels)
