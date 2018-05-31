import pandas as pd
import cPickle
import gzip
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
<<<<<<< HEAD
from sklearn import preprocessing
=======
>>>>>>> 104e75907738d0a3f93fb881a5db97469158e5d2
import matplotlib.pyplot as plt
import csv
import os
####################################################
<<<<<<< HEAD


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
=======
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
>>>>>>> 104e75907738d0a3f93fb881a5db97469158e5d2

########################################################
#TRAINING AND TESTING DATA :
# X - Trainin data
# X_test - testing data
# y - traininglabels
# y_test - testing labels
#X, X_test, y, y_test = train_test_split(T1, y_all,train_size=0.9,random_state=42)


<<<<<<< HEAD
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

=======
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
>>>>>>> 104e75907738d0a3f93fb881a5db97469158e5d2


# #############################################################################
# Look at the results
# Plot the result

<<<<<<< HEAD
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
=======


lw = 2
plt.scatter(y_test, pre, color='navy', lw=lw, label='RBF model')
plt.plot([30, 50], [30, 50], '--k')
plt.xlabel('data')
plt.ylabel('target')
>>>>>>> 104e75907738d0a3f93fb881a5db97469158e5d2
plt.title('Support Vector Regression')
plt.legend()
plt.show()

<<<<<<< HEAD

=======
>>>>>>> 104e75907738d0a3f93fb881a5db97469158e5d2
duration = 1  # second
freq = 440  # Hz
os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors

<<<<<<< HEAD
#model = svm.SVR(kernel='linear',C=1, gamma = 1)
=======
#model = svm.SVC(kernel='linear',C=1, gamma = 1)
>>>>>>> 104e75907738d0a3f93fb881a5db97469158e5d2
#model.fit(X,y)
#s = model.score(X,y)

#predict = model.predict(X_test)
#stest = model.score(X_test,y_test)
#print s
#print stest
