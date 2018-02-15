import pandas as pd
import numpy as np
import os
# import rfcode.helpfunctions as hf
############################################
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_regression
from sklearn import linear_model
import nibabel
from sklearn.feature_selection import SelectPercentile

# long as code thing, numbering, region numbers, T1, T2, Volume (test1.csv)

fields = ['Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']

data = pd.read_csv('/home/avi/Documents/groupprojml/test1.csv',
                    skipinitialspace=True, usecols=fields)
print data.Region
# # Training Data
# subjectpath = ''
# # Testing Data
# datapath = ''

# Reading testing data
# test_data = hf.read_DATA(datapath, readtype)
# training_data = np.asarray(hf.get_subjects(test_data, subjectpath, readtype))
#
# training_labels = []
#
#
# clf = RandomForestClassifier(criterion = "mse", splitter = "best", max_depth = 3,
#                             min_samples_split = 2, max_features = 2)
# clf = clf.fit(training_data, training_labels)
# clf.predict(test_data, training_labels)
