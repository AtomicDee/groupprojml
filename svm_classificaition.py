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
path = '/Users/Mickey/Desktop/Group/Term_Data/'


all_Feat = pd.read_csv(path+'T1_T2_Vol_termandpreterm.csv')
#all_Feat = all_Feat[all_Feat.columns[1:]]
labels = pd.read_csv(path+'Term_labels_termandpreterm.csv')
#labels = labels[labels.columns[2]]

print np.shape(labels)




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
## SEPERATE FEATURES ####################
# RANDOM FOREST
'''
path = '/Users/Mickey/Desktop/Group/Classification_Term_Preterm_above_37_dataset_features.csv'
RF_sf = pd.read_csv(path,names = ['Region','Weight','Type'])
# 50 most important regions - features
print RF_sf
raw_input()
RF_imregion= RF_sf.Region[1:51]
RF_imtype = RF_sf.Type[1:51]
RF_imweight = RF_sf.Weight[1:51]
print RF_imtype
print RF_imweight
print np.shape(RF_imtype)
print np.shape(RF_sf)

path1 = '/Users/Mickey/Desktop/Group/Term_Data/'
T1 = pd.read_csv(path1+'T1_termandpreterm.csv')
print T1
T1 = T1[T1.columns[1:]]
T2 = pd.read_csv(path1+'T2_termandpreterm.csv')
T2 = T2[T2.columns[1:]]
vol = pd.read_csv(path1+'Volume_termandpreterm.csv')
vol= vol[vol.columns[1:]]

print np.shape(T1)

print T1
print np.shape(T2)
print np.shape(vol)
print '##############################'
data_im = pd.DataFrame(index=range(len(T1)), columns=range(50))
data_im = data_im.fillna(0)
print np.shape(data_im)
print RF_imweight
raw_input()
for x in range(len(RF_imregion)):
    reg = RF_imtype[x+1]

    print reg
    print RF_imregion[x+1]
    print int(RF_imregion[x+1])
    if reg == 'T1':
        #for i in range(len(T1)):
        currT1 = T1[T1.columns[int(RF_imregion[x+1])-1]]
        data_im[data_im.columns[x]] = currT1
        print currT1
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
print np.shape(data_im)

data_im = data_im[data_im.columns[:50]]
print np.shape(data_im)
print data_im
print 'HERE'
'''
########################################################
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
print '0.5'


# #############################################################################
'''

X, X_test, y, y_test = train_test_split(x_all, y_all)
print np.shape(X)
print np.shape(X_test)
svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto')
svm.fit(X,y)
pre = svm.predict(X_test)
print '########### DEFAULT ###########'
print np.shape(X)
print np.shape(X_test)
scores = cross_val_score(svm, X, y, cv=5)
curr_mean = np.mean(scores)
print scores
print curr_mean
curr_std = np.std(scores)
print curr_std
print svm.score(X_test,y_test)
print svm.get_params(deep=True)
raw_input()
'''

#####################################################



###############################################################################

gamma = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
kernel = ['linear']
C = [1,10,100]

curr_min = 10000
curr_max = -10000
best_std = -10000
best_score = -1000
best_cross = -1000

all_gamma = []
all_C = []
all_kernel = []
all_scores = []
all_cscores = []
all_std = []
all_all_cscores = []


for x in range(len(kernel)):
     for j in range(len(C)):
        print kernel[x]
        if kernel[x] == 'linear':

            svm = SVC(kernel= 'linear', C=C[j])
            svm.fit(X,y)
            pre = svm.predict(X_test)
            curr_score = svm.score(X_test, y_test)
            curr_cross_score = cross_val_score(svm, x_all, y_all)
            all_all_cscores.append(curr_cross_score[0])
            all_all_cscores.append(curr_cross_score[1])
            all_all_cscores.append(curr_cross_score[2])
            curr_mean = np.mean(curr_cross_score)
            curr_std = np.std(curr_cross_score)
            all_scores.append(curr_score)
            all_cscores.append(curr_mean)
            all_std.append(curr_std)




            if curr_mean > curr_max:
                curr_cross_max = curr_mean
                curr_max = curr_mean

                curr_score_max = curr_score
                curr_max_std = curr_std
                best_C = C[j]
                best_gamma = []
                best_kernel = kernel[x]
                feat_im = svm.coef_

            if curr_mean < curr_min:
                curr_min = curr_mean

                worst_cross = curr_mean
                worst_score = curr_score
                worst_std = curr_std
                worst_C = C[j]
                worst_gamma = []
                worst_kernel = kernel[x]

        if kernel[x] == 'rbf':
            for i in range(len(gamma)):
                svm = SVC(kernel='rbf', C=C[j],gamma=gamma[i])
                svm.fit(X,y)
                pre = svm.predict(X_test)
                curr_score = svm.score(X_test, y_test)
                curr_cross_score = cross_val_score(svm, x_all, y_all)
                all_all_cscores.append(curr_cross_score[0])
                all_all_cscores.append(curr_cross_score[1])
                all_all_cscores.append(curr_cross_score[2])
                curr_mean = np.mean(curr_cross_score)
                curr_std = np.std(curr_cross_score)
                all_scores.append(curr_score)
                all_cscores.append(curr_mean)
                all_std.append(curr_std)


                if curr_mean > curr_max:
                    best_cross = curr_mean
                    curr_max = curr_mean

                    best_score = curr_score
                    best_std = curr_std
                    best_C = C[j]
                    best_gamma = gamma[i]
                    best_kernel = kernel[x]

                if curr_mean < curr_min:
                    curr_min = curr_mean

                    worst_cross = curr_mean
                    worst_score = curr_score
                    worst_std = curr_std
                    worst_C = C[j]
                    worst_gamma = gamma[i]
                    worst_kernel = kernel[x]

print 'Cross Val Scores:'
for i in range(len(all_all_cscores)):
    print all_all_cscores[i]

print 'Cross Val Scores:'
for i in range(len(all_cscores)):
    print all_cscores[i]
print 'STD:'
for i in range(len(all_std)):
    print all_std[i]
print 'Scores:'
for i in range(len(all_scores)):
    print all_scores[i]


print ' '
print '########################################################################'
print 'BEST: '
print best_cross
print best_std
print best_score
print best_kernel
print best_C
print best_gamma
print worst_cross
print worst_std
print worst_score
print worst_kernel
print worst_C
print worst_gamma
print '########################################################################'
print ' '
print feat_im

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
