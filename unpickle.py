import pickle
from sklearn.externals import joblib
import pandas as pd
import csv

clf = joblib.load('DHCP_fmri_datamat.pickle')
# print(clf)

# data = pd.concat(clf)
clf.to_csv('unpickled_data.csv')
