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
path = '/Volumes/Harddrive_MS/Group_Project_Meng/DATA/All_Patients/'
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


#####################################################
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

x_all = data_im
#x_all = all_Feat
y_all = BirthAge


########################################################
#TRAINING AND TESTING DATA :
# X - Trainin data
# X_test - testing data
# y - traininglabels
# y_test - testing labels
#X, X_test, y, y_test = train_test_split(T1, y_all,train_size=0.9,random_state=42)


X, X_test, y, y_test = train_test_split(x_all, y_all,train_size=0.9,random_state=42)


print np.shape(X)
print np.shape(X_test)
print np.shape(y)
print np.shape(y_test)
raw_input()
# #############################################################################


gamma = [1e-7, 1e-6 ,1e-5 ,1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
kernel = ['rbf','linear']
C = [1, 10, 100, 1000]


curr_max = -10000
curr_max_std = -10000
all_best_gamma = []
all_best_C = []
all_best_kernel = []
all_best_scores = []
all_best_cross_scores = []
all_best_std = []


for x in range(len(kernel)):
        for i in range(len(gamma)):
          for j in range(len(C)):
            curr_score = []
            print ' '
            print 'Loop: ', x
            print 'Kernel: ', kernel[x]
            print 'C: ', C[j]
            print 'Gamma: ', gamma[i]
            if kernel[x] == 'linear':
                svr = SVR(kernel= 'linear', C=C[j])
            if kernel[x] == 'rbf':
                svr = SVR(kernel='rbf', C=C[j],gamma=gamma[i])

            svr.fit(X,y)

            pre = svr.predict(X_test)
            curr_score = svr.score(X_test, y_test)
            curr_cross_score = cross_val_score(svr, X_test, y_test)

            curr_mean = np.mean(curr_cross_score)
            curr_std = np.std(curr_cross_score)
            if curr_score > curr_max:
                curr_cross_max = curr_mean
                curr_max = curr_score

                curr_score_max = curr_mean
                curr_max_std = curr_std
                best_C = C[j]
                best_gamma = gamma[i]
                best_kernel = kernel[x]

            all_best_kernel.append(best_kernel)
            all_best_gamma.append(best_gamma)
            all_best_C.append(best_C)
            all_best_scores.append(curr_score_max)
            all_best_cross_scores.append(curr_cross_max)
            all_best_std.append(curr_max_std)


print ' '
print '########################################################################'
print 'The best cross score found was: ', np.mean(all_best_cross_scores)
print 'With a STD of: ', np.mean(all_best_std)
print 'The best score is: ', np.mean(all_best_scores)
print 'The optimal kernel found was: ', all_best_kernel
print 'The optimal number of gamma was found to be: ', np.mean(all_best_gamma)
print 'The best C is: ', np.mean(all_best_C)
print '########################################################################'
print ' '
print '########################################################################'
print 'All cross scores: ', all_best_cross_scores
print 'All STDs: ', all_best_std
print 'All scores: ', all_best_scores
print 'All gamma: ', all_best_gamma
print 'All kernel: ', all_best_kernel
print 'All C: ', all_best_C
print '########################################################################'
print ' '

'''
svr = GridSearchCV(SVR(), parameters, cv=5)

svr.fit(X_scaled,y)
pre = svr.predict(X_test_scaled)
new_s =svr.score(X_test_scaled,y_test)
para_results = svr.cv_results_
para_best = svr.best_params_
est_best = svr.best_estimator_

print para_results
print 'Best Param: ',para_best
print 'Best Est: ',est_best
print 'Best Score: ',svr.best_score_

new_sc = cross_val_score(svr,X_test,y_test)
new_sc = sum(new_sc) / float(len(new_sc))
print 's', new_s, new_sc
'''
###############################################################################



# #############################################################################
# Look at the results
# Plot the result

'''

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
'''
