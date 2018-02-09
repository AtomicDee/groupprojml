import pickle
from sklearn.externals import joblib
import pandas as pd
import csv

clf = joblib.load('dHCP_demographics_filtered30-01-18.pk1')
# print(clf)

# data = pd.concat(clf)
clf.to_csv('unpickled_dHCP_demographics_filtered30-01-18.csv')
