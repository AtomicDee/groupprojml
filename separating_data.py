import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys
<<<<<<< HEAD

'''
this code separates teh data (i.e. t1, t2, volume, patient data, etc.), and
saves them separately - this was mainly used to have different csv files so that
it didn't have to be separated everytime
'''

path = '/home/avi/Desktop/new_normalised_data.csv'
data = pd.read_csv(path)
# Using pandas to read all the data in
=======
>>>>>>> 03e8573aa85a992176222c06f3b59ab1c68dc04b

# path = '/home/avi/Desktop/new_data.csv'
# path = '/home/avi/Desktop/Preterm.csv'
path = '/home/avi/Desktop/Term.csv'
data = pd.read_csv(path)
# Using pandas to read all the data in

<<<<<<< HEAD
=======

>>>>>>> 03e8573aa85a992176222c06f3b59ab1c68dc04b
T1 = []
T2 = []
Volume = []
PatCode = []
SessID = []
BirthAge = []
ScanAge = []
Gender = []
all_feat = []

for row in data.itertuples():
    PatCode.append(pd.DataFrame([row[1]]))
    SessID.append(pd.DataFrame([row[2]]))
    BirthAge.append(pd.DataFrame([row[3]]))
    ScanAge.append(pd.DataFrame([row[4]]))
    Gender.append(pd.DataFrame([row[5]]))
    T1.append(pd.DataFrame([row[6:92]]))
    T2.append(pd.DataFrame([row[92:178]]))
    Volume.append(pd.DataFrame([row[178:267]]))
    all_feat.append(pd.DataFrame([row[6:267]]))
# PatCode = np.ravel( pd.concat( PatCode ) )
# SessID = np.ravel( pd.concat( SessID ) )
# BirthAge = np.ravel( pd.concat( BirthAge ) )
# ScanAge = np.ravel( pd.concat( ScanAge ) )
# Gender = np.ravel( pd.concat( Gender ) )
PatCode = pd.concat( PatCode )
SessID = pd.concat( SessID )
BirthAge = pd.concat( BirthAge )
ScanAge = pd.concat( ScanAge )
Gender = pd.concat( Gender )
T1 = pd.concat(T1)
T2 = pd.concat(T2)
Volume = pd.concat(Volume)
all_feat = pd.concat(all_feat)

print Volume
print np.shape(Volume)

#
# PatCode.to_csv('PatCode.csv')
# SessID.to_csv('SessID.csv')
# BirthAge.to_csv('BirthGA.csv')
# ScanAge.to_csv('ScanGA.csv')
# Gender.to_csv('Gender.csv')
# T1.to_csv('T1.csv')
# T2.to_csv('T2.csv')
# Volume.to_csv('Volume.csv')
# all_feat.to_csv('T1_T2_Vol.csv')
