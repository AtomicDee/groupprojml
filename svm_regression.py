import pandas as pd
import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import csv
import os
####################################################
# import some data to play with
# path = '/Volumes/Harddrive_MS/Group_Project_Meng/DATA/All_Patients/'
path = '/home/avi/Desktop/groupprojml/DATA/All_Patients/'
T1= 'T1.csv'
T1 = pd.read_csv(path+T1)
print np.shape(T1)
T1 = T1[T1.columns[1:86]]
T2= 'T2.csv'
T2 = pd.read_csv(path+T2)
T2 = T2[T2.columns[1:86]]


y_all = T1

Age = 'ScanGA.csv'
all_Feat = 'T1_T2_Vol.csv'
G = 'Gender.csv'
Vol = 'Volume.csv'
BirthAge = 'BirthGA.csv'
# The path where all the data is held

vol = pd.read_csv(path+Vol)
BirthAge = pd.read_csv(path+BirthAge)
BirthAge = BirthAge[BirthAge.columns[1] ]
all_Feat = pd.read_csv(path+all_Feat)
all_Feat = all_Feat[all_Feat.columns[1:]]
#X_all = np.column_stack((T1v,T2v))
#y_all = pd.read_csv(path+Age)

####################################################
#RANDOM FOREST

path_feat = '/home/avi/Desktop/'
# ''
# t1f = 'T1_features_sorted.csv'
# t2f = 'T2_features.csv'
# volf = 'Vol_features.csv'
# T1f = pd.read_csv(path_feat+t1f)
# ''
#####################################################
#ADABOOST

sorted_feat = 'adaboost_sortedfeat.csv'
pa = path_feat + sorted_feat
ada_sf = pd.read_csv(path_feat+sorted_feat,names = ['Region','Weight','Type','W2'])
# 50 most important regions - features
ada_imregion= ada_sf.Region[:51]
ada_imtype = ada_sf.Type[:51]

data_im = pd.DataFrame(index=range(503), columns=range(50))
data_im = data_im.fillna(0)

for x in range(len(ada_imregion)-1):
    reg = ada_imtype[x+1]
    if reg == 't1':
        #for i in range(len(T1)):
        currT1 = T1[T1.columns[int(ada_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currT1
        #pd.data_im.concat([data_im, currT1], axis=1)

    if reg == 't2':
        #for i in range(len(T1)):
        currT2 = T2[T2.columns[int(ada_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currT2

        #pd.data_im.concat([data_im, currT2], axis=1)
    if reg == 'vol':
    #for i in range(len(T1)):
        currVol = vol[vol.columns[int(ada_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currVol
        #pd.data_im.concat([data_im, currVol], axis=1)

########################################################

x_all = data_im
y_all = BirthAge

########################################################
#TRAINING AND TESTING DATA :
# X - Trainin data
# X_test - testing data
# y - traininglabels
# y_test - testing labels
#X, X_test, y, y_test = train_test_split(T1, y_all,train_size=0.9,random_state=42)


X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.9,random_state=42)



''
# print len(X)
# print len(y)
# y = np.ravel(y)
# c = [1e-10, 1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 9e-7, 1e-6, 2e-6, 3e-6, 4e-6, 7e-6, 8e-6]
# gam = [1e1, 1e2, 1e3,2e3,4e3,5e3,7e3,9e3]
# max_sc_lin = 0
# max_sc_rbf = 0
# max_sc_poly = 0
# ind_i_rbf = 0
# ind_j_rbf = 0
# for i in range(len(gam)):
#     for j in range(len(c)):
#         print c[j]
#         print gam[i]
#         svr_rbf = SVR(kernel='rbf',C=c[j],gamma = gam[i])
#         svr_rbf.fit(X,y)
#         pre_rbf = svr_rbf.predict(X_test)
#
#         #scores_y1 = cross_val_score(svr_rbf, X_test, y_test)
#         #print scores_y1.mean(), scores_y1.std()
#
#         sc_rbf = svr_rbf.score(X_test,y_test)
#         print sc_rbf
#
#         if sc_rbf > max_sc_rbf:
#             max_sc_rbf = sc_rbf
#             print 'R'
#             print max_sc_rbf
#             ind_i_rbf = i
#             ind_j_rbf = j



#
# print 'R',ind_i_rbf
# print ind_j_rbf
# print max_sc_rbf

# #############################################################################

svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e5,1e6,1e7,1e8,1e9],
                               "gamma": np.logspace(-10, 10, 20)})

svr.fit(X,y)
pre = svr.predict(X_test)
new_s =svr.score(X_test,y_test)
para_results = svr.cv_results_
para_best = svr.best_params_
# para_score = svr.
print para_results
print para_best

new_sc = cross_val_score(svr,X_test,y_test)
new_sc = sum(new_sc) / float(len(new_sc))
print 's', new_s, new_sc
###############################################################################

'''
# svrl = GridSearchCV(SVR(kernel='linear'), cv=5,
#                    param_grid={"C": [1,10,100,1000])})
# svrl.fit(X,y)
# prel = svrl.predict(X_test)
# new_sl =svrl.score(X_test,y_test)
# new_scl = cross_val_score(svrl,X_test,y_test)
# new_scl = sum(new_scl) / float(len(new_scl))
# print 'sline', new_sl, new_scl
'''


# #############################################################################
# Look at the results
# Plot the result



lw = 2
plt.scatter(y_test, pre, color='navy', lw=lw, label='RBF model')
plt.plot([30, 50], [30, 50], '--k')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

duration = 1  # second
freq = 440  # Hz
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors

#model = svm.SVC(kernel='linear',C=1, gamma = 1)
#model.fit(X,y)
#s = model.score(X,y)

#predict = model.predict(X_test)
#stest = model.score(X_test,y_test)
#print s
#print stest
