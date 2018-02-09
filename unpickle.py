import pickle
from sklearn.externals import joblib
import pandas as pd
import csv

clf = joblib.load('dHCP_demographics_filtered30-01-18.pk1')
# print(clf)

# data = pd.concat(clf)
<<<<<<< HEAD
clf.to_csv('dHCP_demographics.csv')
=======
clf.to_csv('unpickled_dHCP_demographics_filtered30-01-18.csv')
>>>>>>> cb2ad6f54961d62fbb58dd4e2f8227b38a2ab695
