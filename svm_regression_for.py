import pandas as pd
import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
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

labels = pd.read_csv(path+'BirthGA.csv')
labels = labels[labels.columns[2]]
print all_Feat
print labels

print np.shape(all_Feat)
print np.shape(labels)

#####################################################
## SEPERATE FEATURES ####################
# RANDOM FOREST
'''
path = '/Users/Mickey/Desktop/Group/ADA_FULL_REG.csv'
RF_sf = pd.read_csv(path,names = ['Region','Weight','Type'])
# 50 most important regions - features
print RF_sf
raw_input()
RF_imregion= RF_sf.Region[1:66]
RF_imtype = RF_sf.Type[1:66]
RF_imweight = RF_sf.Weight[1:66]
print RF_imtype
print RF_imweight
print np.shape(RF_imtype)
print np.shape(RF_sf)

path1 = '/Users/Mickey/Desktop/Group/All_Data/'
T1 = pd.read_csv(path1+'T1.csv')
print T1
T1 = T1[T1.columns[1:]]
T2 = pd.read_csv(path1+'T2.csv')
T2 = T2[T2.columns[1:]]
vol = pd.read_csv(path1+'Volume.csv')
vol= vol[vol.columns[1:]]

print np.shape(T1)

print T1
print np.shape(T2)
print np.shape(vol)
print '##############################'
data_im = pd.DataFrame(index=range(len(T1)), columns=range(65))
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

data_im = data_im[data_im.columns[:65]]
print np.shape(data_im)
print data_im
print 'HERE'
###############################
'''
#x_all = data_im
x_all = all_Feat
y_all = labels
print x_all
print y_all
print np.shape(x_all)
print np.shape(y_all)
raw_input()

########################################################
#TRAINING AND TESTING DATA :
# X - Trainin data
# X_test - testing data
# y - traininglabels
# y_test - testing labels
#X, X_test, y, y_test = train_test_split(T1, y_all,train_size=0.9,random_state=42)


X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.5,random_state = 42)
print '0.5'

y = np.ravel(y)
y_test = np.ravel(y_test)
print y_test
raw_input()
# #############################################################################
'''
X, X_test, y, y_test = train_test_split(x_all, y_all)
print np.shape(X)
print np.shape(X_test)
svm = SVR(C=1.0, kernel='linear', degree=3, gamma='auto')
svm.fit(X,y)
pre = svm.predict(X_test)
print '########### DEFAULT ###########'
print np.shape(X)
print np.shape(X_test)
scores = cross_val_score(svm, x_all, y_all)
curr_mean = np.mean(scores)
print scores
print curr_mean
curr_std = np.std(scores)
print curr_std
print svm.score(X_test,y_test)
print svm.get_params(deep=True)
raw_input()
'''
###############################################################################

gamma = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2,1e3]
kernel = ['rbf']
C = [1, 10, 100,1000]

curr_max = -10000
curr_min = 10000
all_gamma = []
all_C = []
all_kernel = []
all_scores = []
all_all_cscores = []
all_cscores = []
all_std = []

for x in range(len(kernel)):
     for j in range(len(C)):
        if kernel[x] == 'linear':
            if C[j] < 1000:
                print kernel[x], C[j]
                svm = SVR(kernel= kernel[x], C=C[j])
                svm.fit(X,y)
                pre = svm.predict(X_test)

                curr_score = svm.score(X_test, y_test)
                curr_cross_score = cross_val_score(svm, x_all, y_all)
                curr_mean = np.mean(curr_cross_score)
                curr_std = np.std(curr_cross_score)
                print curr_score
                print curr_mean
                print curr_std
                all_all_cscores.append(curr_cross_score[0])
                all_all_cscores.append(curr_cross_score[1])
                all_all_cscores.append(curr_cross_score[2])
                all_scores.append(curr_score)
                all_cscores.append(curr_mean)
                all_std.append(curr_std)

                if curr_mean > curr_max:
                    curr_max = curr_mean

                    best_cross = curr_mean
                    best_score = curr_score
                    best_std = curr_std
                    best_C = C[j]
                    best_gamma = []
                    best_kernel = kernel[x]

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
                print kernel[x], C[j], gamma[i]
                svm = SVR(kernel='rbf', C=C[j],gamma=gamma[i])
                svm.fit(X,y)
                pre = svm.predict(X_test)

                curr_score = svm.score(X_test, y_test)
                curr_cross_score = cross_val_score(svm, x_all, y_all)
                all_all_cscores.append(curr_cross_score[0])
                all_all_cscores.append(curr_cross_score[1])
                all_all_cscores.append(curr_cross_score[2])
                curr_mean = np.mean(curr_cross_score)
                curr_std = np.std(curr_cross_score)
                curr_mean = np.mean(curr_cross_score)
                curr_std = np.std(curr_cross_score)
                print curr_score
                print curr_mean
                print curr_std
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

                if curr_mean< curr_min:
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
print 'Mean Cross Val Scores:'
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
print 'BEST AND WORST'
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



###############################################################################



# #############################################################################
# Look at the results
# Plot the result

predw = []
predb = []
### BEST
svm_best = SVR(kernel=best_kernel, C=best_C,gamma=best_gamma)
svm_best.fit(X,y)
pre_best = svm_best.predict(X_test)
lw = 2




#WORST
svm_worst = SVR(kernel=worst_kernel, C=worst_C,gamma=worst_gamma)
svm_worst.fit(X,y)
pre_worst = svm_worst.predict(X_test)
lw = 2


plt.subplot(1,2,1)
plt.scatter(y_test, pre_best, color='navy', lw=lw)
plt.plot([30, 50], [30, 50], '--k')
plt.xlabel('Data')
plt.ylabel('Predicted')
plt.title('Support Vector Regression')
plt.legend()

plt.subplot(1,2,2)
plt.scatter(y_test, pre_worst, color='navy', lw=lw)
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

#model = svm.SVR(kernel='linear',C=1, gamma = 1)
#model.fit(X,y)
#s = model.score(X,y)

#predict = model.predict(X_test)
#stest = model.score(X_test,y_test)
#print s
#print stest
