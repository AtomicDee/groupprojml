import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys

path = '/home/avi/Desktop/new_data.csv'

data = pd.read_csv(path)

preterm = []
term = []
count = 0

for row in data.itertuples():
    if row[4] < 37:
        preterm.append(pd.DataFrame([row]))
    else:
        term.append(pd.DataFrame([row]))
# remove any subject data below 37 weeks

preterm = pd.concat(preterm)
term = pd.concat(term)

# preterm.to_csv('Preterm.csv')
# term.to_csv('Term.csv')
#save the new term/preterm data
