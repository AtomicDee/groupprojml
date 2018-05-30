import pandas as pd
import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import csv
import os
####################################################


# import some data to play with
path = '/Users/Mickey/Desktop/Group/All_Data/'


all_Feat = pd.read_csv(path+'T1_T2_Vol.csv')
all_Feat = all_Feat[all_Feat.columns[1:]]
labels = pd.read_csv(path+'Gender.csv')
labels = labels[labels.columns[2]]

Vol = 'Volume.csv'
BirthAge = 'BirthGA.csv'
# The path where all the data is held

#vol = pd.read_csv(path+Vol)
#BirthAge = pd.read_csv(path+BirthAge)
#BirthAge = BirthAge[BirthAge.columns[1] ]
#all_Feat = pd.read_csv(path+all_Feat)
#all_Feat = all_Feat[all_Feat.columns[1:]]
#X_all = np.column_stack((T1v,T2v))
#y_all = pd.read_csv(path+Age)


print all_Feat
print np.shape(all_Feat)
print labels
raw_input()
#####################################################
'''
#ADABOOST
path_feat = '/Volumes/Harddrive_MS/Group_Project_Meng/Features/'
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
'''
########################################################
#####################################################
'''
# RANDOM FOREST
path_feat = '/Volumes/Harddrive_MS/Group_Project_Meng/RF_Feature_importances/'
sorted_feat = 'sortedfeat_RF_Regr.csv'
pa = path_feat + sorted_feat
RF_sf = pd.read_csv(path_feat+sorted_feat,names = ['Region','Weight','Type'])
# 50 most important regions - features
RF_imregion= RF_sf.Region[1:51]
RF_imtype = RF_sf.Type[1:51]
RF_imweight = RF_sf.Weight[1:51]
data_im = pd.DataFrame(index=range(503), columns=range(50))
data_im = data_im.fillna(0)

print RF_imweight
raw_input()
for x in range(len(RF_imregion)-1):
    reg = RF_imtype[x+1]
    if reg == 'T1':
        #for i in range(len(T1)):
        currT1 = T1[T1.columns[int(RF_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currT1
        #pd.data_im.concat([data_im, currT1], axis=1)

    if reg == 'T2':
        #for i in range(len(T1)):
        currT2 = T2[T2.columns[int(RF_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currT2

        #pd.data_im.concat([data_im, currT2], axis=1)
    if reg == 'Vol':
    #for i in range(len(T1)):
        currVol = vol[vol.columns[int(RF_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currVol
        #pd.data_im.concat([data_im, currVol], axis=1)


print data_im


print np.shape(data_im)
print np.shape(RF_imweight)
'''
########################################################

#x_all = data_im
x_all = all_Feat
y_all = labels


########################################################
#TRAINING AND TESTING DATA :
# X - Trainin data
# X_test - testing data
# y - traininglabels
# y_test - testing labels
#X, X_test, y, y_test = train_test_split(T1, y_all,train_size=0.9,random_state=42)


X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.5,random_state=42)


# #############################################################################
'''

X, X_test, y, y_test = train_test_split(x_all, y_all)
print np.shape(X)
print np.shape(X_test)
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
svm.fit(X,y)
pre = svm.predict(X_test)
print '########### DEFAULT ###########'
print 'Score : ', svm.score(X_test,y_test)
scores = cross_val_score(svm, x_all, y_all, cv=5)
print 'Cross val Scores : ', scores
curr_mean = np.mean(scores)
print 'Mean cvs : ', curr_mean
curr_std = np.std(scores)
print 'Std cvs : ', curr_std

print svm.get_params(deep=True)
raw_input()
'''
###############################################################################

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100,1000],'gamma' :[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]}
svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(x_all, y_all)
print 'CV = 3 cross fold'
print clf.best_params_
print clf.score
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print clf.score(X_test,y_test)

y_pred = clf.predict(X_test)
##############################


parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10,100,1000],'gamma' :[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]}
svr = SVC()
clf = GridSearchCV(svr, parameters, cv = 5)
clf.fit(x_all, y_all)
print 'CV = 5 cross fold'
print clf.best_params_
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print clf.score(X_test,y_test)
curr_cross_score = cross_val_score(clf, X_test, y_test, cv=5)
curr_mean = np.mean(curr_cross_score)
curr_std = np.std(curr_cross_score)

print curr_mean
y_pred = clf.predict(X_test)
###############################################################################



# #############################################################################
# Look at the results
# Plot the result

'''

lw = 2
plt.scatter(y_test, pre, color='navy', lw=lw, label='RBF model')
plt.plot([30, 50], [30, 50], '--k')
plt.xlabel('Data')
plt.ylabel('Predicted')
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
'''
