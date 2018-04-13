import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import nibabel
import csv
import sys

# path = '/home/avi/Desktop/new_data.csv'
# path = '/home/avi/Desktop/Preterm.csv'
# path = '/home/avi/Desktop/Term.csv'
path = '/home/avi/Desktop/test_data_only/all_data'
term_path = '/home/avi/Desktop/test_data_only/Term_Data'
preterm_path = '/home/avi/Desktop/test_data_only/New_Preterm_Data'

preterm_ScanAge = pd.read_csv(preterm_path + '/ScanGA.csv')
preterm_BirthAge = pd.read_csv(preterm_path + '/BirthGA.csv')
preterm_all_feat = pd.read_csv(preterm_path + '/T1_T2_Vol.csv')
preterm_Gender = pd.read_csv(preterm_path + '/Gender.csv')
preterm_PatCode = pd.read_csv(preterm_path + '/PatCode.csv')
preterm_T1 = pd.read_csv(preterm_path + '/T1.csv')
preterm_T2 = pd.read_csv(preterm_path + '/T2.csv')
preterm_Vol = pd.read_csv(preterm_path + '/Volume.csv')
preterm_SessID = pd.read_csv(preterm_path + '/SessID.csv')
preterm_labels = pd.read_csv(preterm_path + '/Term_labels.csv')
preterm_all_features = pd.read_csv(preterm_path + '/ALL_FEATURES.csv')

term_ScanAge = pd.read_csv(term_path + '/ScanGA.csv')
term_BirthAge = pd.read_csv(term_path + '/BirthGA.csv')
term_all_feat = pd.read_csv(term_path + '/T1_T2_Vol.csv')
term_Gender = pd.read_csv(term_path + '/Gender.csv')
term_PatCode = pd.read_csv(term_path + '/PatCode.csv')
term_T1 = pd.read_csv(term_path + '/T1.csv')
term_T2 = pd.read_csv(term_path + '/T2.csv')
term_Vol = pd.read_csv(term_path + '/Volume.csv')
term_SessID = pd.read_csv(term_path + '/SessID.csv')
term_labels = pd.read_csv(term_path + '/Term_labels.csv')
term_all_features = pd.read_csv(term_path + '/ALL_FEATURES.csv')

all_ScanAge = []
all_BirthAge = []
all_feat_not_scan = []
all_Gender = []
all_PatCode = []
all_T1 = []
all_T2 = []
all_Vol = []
all_SessID = []
all_term_labels = []
all_features = []

all_ScanAge.append(pd.DataFrame(preterm_ScanAge))
all_ScanAge.append(pd.DataFrame(term_ScanAge))

all_BirthAge.append(pd.DataFrame(preterm_BirthAge))
all_BirthAge.append(pd.DataFrame(term_BirthAge))

all_feat_not_scan.append(pd.DataFrame(preterm_all_feat))
all_feat_not_scan.append(pd.DataFrame(term_all_feat))

all_Gender.append(pd.DataFrame(preterm_Gender))
all_Gender.append(pd.DataFrame(term_Gender))

all_PatCode.append(pd.DataFrame(preterm_PatCode))
all_PatCode.append(pd.DataFrame(term_PatCode))

all_T1.append(pd.DataFrame(preterm_T1))
all_T1.append(pd.DataFrame(term_T1))

all_T2.append(pd.DataFrame(preterm_T2))
all_T2.append(pd.DataFrame(term_T2))

all_Vol.append(pd.DataFrame(preterm_Vol))
all_Vol.append(pd.DataFrame(term_Vol))

all_SessID.append(pd.DataFrame(preterm_SessID))
all_SessID.append(pd.DataFrame(term_SessID))

all_term_labels.append(pd.DataFrame(preterm_labels))
all_term_labels.append(pd.DataFrame(term_labels))

all_features.append(pd.DataFrame(preterm_all_features))
all_features.append(pd.DataFrame(term_all_features))

all_ScanAge = pd.concat(all_ScanAge)
all_BirthAge = pd.concat(all_BirthAge)
all_feat_not_scan = pd.concat(all_feat_not_scan)
all_Gender = pd.concat(all_Gender)
all_PatCode = pd.concat(all_PatCode)
all_T1 = pd.concat(all_T1)
all_T2 = pd.concat(all_T2)
all_Vol = pd.concat(all_Vol)
all_SessID = pd.concat(all_SessID)
all_term_labels = pd.concat(all_term_labels)
all_features = pd.concat(all_features)

# for row in all_BirthAge.itertuples():
#     if row[2] < 37:
#         all_term_labels.append(pd.DataFrame(['Preterm']))
#     elif row[2] >= 37:
#         all_term_labels.append(pd.DataFrame(['Term']))
#     else:
#         print('HOW DID THIS EVEN HAPPEN')
#         print(row)
#

all_PatCode.to_csv(os.path.join(path,'PatCode.csv'))
all_SessID.to_csv(os.path.join(path,'SessID.csv'))
all_BirthAge.to_csv(os.path.join(path,'BirthGA.csv'))
all_ScanAge.to_csv(os.path.join(path,'ScanGA.csv'))
all_term_labels.to_csv(os.path.join(path, 'Term_labels.csv'))
all_Gender.to_csv(os.path.join(path,'Gender.csv'))
all_T1.to_csv(os.path.join(path,'T1.csv'))
all_T2.to_csv(os.path.join(path,'T2.csv'))
all_Vol.to_csv(os.path.join(path,'Volume.csv'))
all_feat_not_scan.to_csv(os.path.join(path, 'T1_T2_Vol.csv'))
all_features.to_csv(os.path.join(path, 'ALL_FEATURES.csv'))

#
