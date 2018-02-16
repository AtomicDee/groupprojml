import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
# from sklearn import export_graphviz
import nibabel
import csv

# long as code thing, numbering, region numbers, T1, T2, Volume (test1.csv)

fields = ['Pat ID', 'Session ID', 'Birth Age', 'Scan Age', 'Gender', 'Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']
# Select which columns to load

# path = '/Volumes/Harddrive_MS/Group_Project_Meng/groupprojml/finaldata.csv'
path = '/home/avi/Desktop/finaldata.csv'
# The path where all the data is held

data = pd.read_csv(path, skipinitialspace=True, usecols=fields)
# Using pandas to read all the data in

sessID = data['Session ID']
birth = data['Birth Age']
scan = data['Scan Age']
gender = data['Gender']
region = data['Region']
T1 = data['T1 Average Intensity']
T2 = data['T2 Average Intensity']
Volume = data['Volume']
patID = data['Pat ID']
# Assigning variables

new_data = []
df = []
initrow = []
row = [None]*268
curr_sess = currT1 = currT2 = currVol = []
for i in range(1):
    x = 1
    row[0] = patID[x]
    row[1] = sessID[x]
    row[2] = gender[x]
    row[3] = birth[x]
    row[4]= scan[x]
    # initrow = [currID, currSess, currGen, currBirth, currScan]

    for j in range(87):

        row[j+5] = T1[j*x]
        row[j+93] = T2[j*x]
        row[j+181] = Volume[j*x]

    print row
    data.append(pd.DataFrame(row))
    raw_input()

    # np.concatenate((new_data, row), axis=1)
    # new_data.append(row)
    #df.append(pd.DataFrame(row))
    initrow = row = curr_sess = currT1 = currT2 = currVol = []
    x += 87


data.to_csv('New_formatted_data.csv')

# training_data = T2.values.reshape(-1, 1)
# training_labels = gender
#
# clf = RandomForestClassifier()
# # clf = RandomForestClassifier(criterion = "mse", splitter = "best", max_depth = 3,
# #                             min_samples_split = 2, max_features = 2)
# clf = clf.fit(training_data, training_labels)
#
# tree.export_graphviz(clf)

# clf.predict(test_data, training_labels)
