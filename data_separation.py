import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys

# path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
path = '/home/avi/Desktop/data with zero columns/'
path2 = '/home/avi/Desktop/data with zero columns/new/'
filename = 'all_data_no_dups_all_above_37.csv'
# The path where all the data is held

data = pd.read_csv(path+filename)
# Using pandas to read all the data in
# print data, data.shape
T1 = []
T2 = []
Volume = []
PatCode = []
SessID = []
BirthAge = []
ScanAge = []
Gender = []
Term = []
T1_T2_Vol = []
# all_features = []

for row in data.itertuples():
    PatCode.append(pd.DataFrame([row[1]]))
    SessID.append(pd.DataFrame([row[2]]))
    BirthAge.append(pd.DataFrame([row[3]]))
    ScanAge.append(pd.DataFrame([row[4]]))
    Term.append(pd.DataFrame([row[5]]))
    Gender.append(pd.DataFrame([row[6]]))
    T1.append(pd.DataFrame([row[7:93]]))
    T2.append(pd.DataFrame([row[93:179]]))
    Volume.append(pd.DataFrame([row[179:268]]))
    T1_T2_Vol.append(pd.DataFrame([row[7:268]]))
    # all_features.append(pd.DataFrame([row[7:268], row[4]]))

        # T1.append(pd.DataFrame([row[6:92]]))
        # T2.append(pd.DataFrame([row[92:178]]))
        # Volume.append(pd.DataFrame([row[178:267]]))
        # all_feat.append(pd.DataFrame([row[6:267]]))
    # print np.shape(T1)
    # print np.shape(T2)
    # print np.shape(Volume)
    # raw_input()

PatCode = pd.concat( PatCode )
SessID = pd.concat( SessID )
BirthAge = pd.concat( BirthAge )
ScanAge = pd.concat( ScanAge )
Gender = pd.concat( Gender )
Term = pd.concat( Term )

T1 = pd.concat(T1)
# print 'shape t1 : ', T1.shape
T2 = pd.concat(T2)
# print 'shape t2 : ', T2.shape
Volume = pd.concat(Volume)
# print 'shape vol : ', Volume.shape

T1_T2_Vol = pd.concat(T1_T2_Vol)

# print BirthAge
# all_features = pd.concat(all_features)
# Save all these separately
# PatCode.to_csv(os.path.join(path,'PatCode.csv'))
# SessID.to_csv(os.path.join(path,'SessID.csv'))
# BirthAge.to_csv(os.path.join(path,'BirthAge.csv'))
# ScanAge.to_csv(os.path.join(path,'ScanAge.csv'))
# Gender.to_csv(os.path.join(path,'Gender.csv'))
# T1.to_csv(os.path.join(path,'T1.csv'))
# T2.to_csv(os.path.join(path,'T2.csv'))
# Volume.to_csv(os.path.join(path,'Volume.csv'))

PatCode.to_csv(os.path.join(path2,'PatCode.csv'))
SessID.to_csv(os.path.join(path2,'SessID.csv'))
BirthAge.to_csv(os.path.join(path2,'BirthGA.csv'))
ScanAge.to_csv(os.path.join(path2,'ScanGA.csv'))
Term.to_csv(os.path.join(path2, 'Term_labels.csv'))
Gender.to_csv(os.path.join(path2,'Gender.csv'))
T1.to_csv(os.path.join(path2,'T1.csv'))
T2.to_csv(os.path.join(path2,'T2.csv'))
Volume.to_csv(os.path.join(path2,'Volume.csv'))
T1_T2_Vol.to_csv(os.path.join(path2,'ALL_FEATURES.csv'))
