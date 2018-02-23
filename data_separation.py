import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys

path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
filename = 'Reformatted_data.csv'
# The path where all the data is held

data = pd.read_csv(path+filename)
# Using pandas to read all the data in

T1 = []
T2 = []
Volume = []
PatCode = []
SessID = []
BirthAge = []
ScanAge = []
Gender = []

for row in data.itertuples():
    PatCode.append(pd.DataFrame([row[2]]))
    SessID.append(pd.DataFrame([row[3]]))
    BirthAge.append(pd.DataFrame([row[4]]))
    ScanAge.append(pd.DataFrame([row[5]]))
    Gender.append(pd.DataFrame([row[6]]))
    T1.append(pd.DataFrame([row[7:94]]))
    T2.append(pd.DataFrame([row[95:182]]))
    Volume.append(pd.DataFrame([row[183:270]]))

PatCode = pd.concat( PatCode )
SessID = pd.concat( SessID )
BirthAge = pd.concat( BirthAge )
ScanAge = pd.concat( ScanAge )
Gender = pd.concat( Gender )

T1 = pd.concat(T1)
T2 = pd.concat(T2)
Volume = pd.concat(Volume)

# Save all these separately
PatCode.to_csv(os.path.join(path,'PatCode.csv'))
SessID.to_csv(os.path.join(path,'SessID.csv'))
BirthAge.to_csv(os.path.join(path,'BirthAge.csv'))
ScanAge.to_csv(os.path.join(path,'ScanAge.csv'))
Gender.to_csv(os.path.join(path,'Gender.csv'))
T1.to_csv(os.path.join(path,'T1.csv'))
T2.to_csv(os.path.join(path,'T2.csv'))
Volume.to_csv(os.path.join(path,'Volume.csv'))
