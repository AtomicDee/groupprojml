# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:41:53 2017

@author: emmar
"""

import numpy as np
import pickle
import pandas as pd
import scipy.io as sio
import nibabel
import copy
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn import svm
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def read_DATA(path,datatype):
    # read data files of different input types '.mat' = Parcellation eval paper format, '.func' is raw HCP multimodal features and labels
    # HCP_avfeatures is HCP multimodal featureset after processing all subjects into one data matrix
    if datatype=='.mat':
        DATA=sio.loadmat()
    elif datatype=='.func.gii':
        print('run averageing')
        trainingDATA=average_multimodal_parcellation_features_all(path)
    elif datatype=='HCP_avfeatures':
        file=open(path,'rb')
        trainingDATA=pickle.load(file)
        file.close()
        #trainingDATA=np.load(path)
        
    return trainingDATA

def get_subjects(DATA,datatype):
    if datatype=='.mat':
        subjects=DATA.get('subjects')
    else:
        subjects=DATA['subjid']
            
    return subjects


def get_data(DATA, datatype):
    if datatype == '.mat':
        trainingDATA,indices=convert_mat(DATA)
    else:
        subjects = DATA['features']

    return subjects
    
def convert_mat(MAT):
    # convert '.mat' format into feature vectors
    DATA=MAT.get('connectivity')
    vectoriseddata=[]
    print(len(MAT.get('subjects')),np.arange(len(MAT.get('subjects'))))
    for index in np.arange(len(MAT.get('subjects'))):
        vectoriseddata.append(get_uppertriangle(DATA[:,:,index]))
        
    return np.asarray(vectoriseddata)
    
def get_uppertriangle(subjectmatrix):
    # remove symmetric results
    iu1 = np.triu_indices(360) # get indices for upper triangular values
    return subjectmatrix[iu1],iu1
    
def average_multimodal_parcellation_features(featurefunc,labelfunc,labelrange):
    # average the surface results within each parcel and save in a data matrix
    features=np.zeros((len(labelrange)*featurefunc.numDA))
    average_func=np.zeros((labelfunc.darrays[0].data.shape[0],featurefunc.numDA))
    
    for index,val in enumerate(labelrange):
       # print(index,val,len(labelrange),featurefunc.numDA)
        x=np.where(labelfunc.darrays[0].data ==val)
        if x[0].shape != (0,):
            for i in np.arange(featurefunc.numDA):
                features[index*featurefunc.numDA+i]=np.mean(featurefunc.darrays[i].data[x[0]])
                average_func[[x],i]=np.mean(featurefunc.darrays[i].data[x[0]])
                if np.isnan(features[index*featurefunc.numDA+i]) :
                    print('isnan',index,val,x[0].shape,np.where(np.isnan(featurefunc.darrays[i].data[x[0]])))
        else:
            for i in np.arange(featurefunc.numDA):
                features[index*featurefunc.numDA+i]=0
                average_func[[x],i]=0
                
    return features,average_func
    
def average_multimodal_parcellation_features_all(files):
    dataset=pd.read_csv(files,sep=' ')
    #giftipaths=pd['features'] #np.genfromtxt(files,dtype=str);
   # labelpaths = pd['labels']

    trainingDATA=[]
    trainingfunc=[]

    for i in np.arange(len(dataset)):
        featuresLR=[]
        for hemi in ['L', 'R']:
            featurepath=dataset['features'][i].replace('%hemi%',hemi)
            labelpath=dataset['labels'][i].replace('%hemi%',hemi)
            print(dataset['features'][i],featurepath)

            funcdata=nibabel.load(featurepath)
            labeldata=nibabel.load(labelpath)
            #print(hemi, np.unique(labeldata.darrays[0].data))
            features,avfunc=average_multimodal_parcellation_features(funcdata,labeldata,np.arange(1,181))
            np.savetxt(featurepath.replace('func.gii','average.txt'),avfunc)
      
            if hemi=='L':
                featuresLR=features
            else:
                featuresLR2=np.concatenate((featuresLR,features))
        trainingDATA.append(featuresLR2)
    trainingDATA=np.asarray(trainingDATA)
    alldata={}
    alldata['subjid'] = dataset['subjid']
    alldata['features'] = trainingDATA

    #alldata = pd.DataFrame(trainingDATA)
    return alldata
        
def map_feature_importances_back_to_image_space(flist,indices,labelfunc):
    
    mappings=np.zeros[(len(flist),labelfunc.shape[0])]
    
    for i in np.arange(len(flist)):
        regions=indices[flist[i]]
        x=np.where(labelfunc.darrays[0].data ==regions[0])
        y=np.where(labelfunc.darrays[0].data ==regions[1])
        mappings[i,x]=1;
        mappings[i,y]=2;
        
    return mappings
    
def map_feature_importances_back_to_image_space_HCP(numfeatures,numregions,importances,labelfuncL, labelfuncR):
    # map feature importances back to image domain from HCP multimodal features
    # fmask is the mask output from an initial feature selection step used to remove noisy features
    # importances is feature importances from final random forest 
    # labelfunc (one fore each hemisphere) is a gifti that contains label membership for regions across the surface
    

    mappingsL = np.zeros(labelfuncL.darrays[0].data.shape)
    mappingsR = np.zeros(labelfuncL.darrays[0].data.shape)
    importantfeatures=np.zeros((len(importances),2))
    for index,val in enumerate(importances):


        true_index=val
        region=int(true_index/numfeatures)+1 # add one because background is not included
        opt_feature=true_index -region*numfeatures
        importantfeatures[index,0]=region
        importantfeatures[index,1]=opt_feature
     #   print(index,true_index,region,opt_feature,len(importances))

        if region<=180:
            # then it is a left hemisphere feature     
            x=np.where(labelfuncR.darrays[0].data ==region)[0]
         #   print('R',x.shape)
            mappingsL[x]=index;
            #mappingsR.append(mappings)
            #labeltmpR.darrays[0].data=mappings[x]
            #labelRout.add_gifti_data_array(nibabel.gifti.GiftiDataArray(mappings[x]))
        else:
            x=np.where(labelfuncL.darrays[0].data ==region)[0]
      #      print('L',x.shape)

            mappingsR[x] = index;
            #print(np.unique(mappings),np.where(mappings>0)[0].shape)
            #mappingsL.append(mappings)

    #print(np.unique(np.asarray(mappingsL)), np.where(np.asarray(mappingsL) > 0)[0].shape)
    return mappingsL,mappingsR,importantfeatures
    
def return_kbest_features(features,labels,features_test,perc,method):
    # use statistical tests to remove noisiest features
     if method=='regression':
         print('kbest: regression',perc)
         kbest=SelectPercentile(score_func=f_regression, percentile=perc)
     else:
         #print('kbest: classification')
         kbest=SelectPercentile(percentile=perc)

     featuresperc = kbest.fit_transform(features, labels)
     return featuresperc,kbest.transform(features_test)
     
def optimise_feature_selection(features,labels,method,rand,optalpha=10):
    #optimise parameters for feature selection by observing performance of simple linear classifer/regressor:
    if method=='regression':
        transform = SelectPercentile(f_regression)
        model=RandomForestRegressor(n_estimators=1000,random_state=rand,n_jobs=-1)
        clf = Pipeline([('anova', transform), ('ridge', model)])
    else:
        transform = SelectPercentile()

        clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])

            
        
    score_means = list()
    score_stds = list()
    percentiles = (0.001, 0.005, 0.01 ,0.1, 0.5, 1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
         clf.set_params(anova__percentile=percentile)
     # Compute cross-validation score using 1 CPU
         this_scores = cross_val_score(clf, features, labels, cv=5,n_jobs=1)
         score_means.append(this_scores.mean())
         score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds))

    plt.title(
               'Performance of the simple classifier/regressor varying the percentile of features selected')
    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')

    plt.axis('tight')
    plt.show()

    print(max(score_means),score_means.index(max(score_means)),np.where(score_means==np.max(score_means)))             
    return percentiles[score_means.index(max(score_means))]

def optimise_feature_selection_stage2(indices,features,labels,method,rand,optalpha=10):
    #optimise parameters for feature selection by observing performance of simple linear classifer/regressor:
    if method=='regression':
        
        model=RandomForestRegressor(n_estimators=1000,random_state=rand,n_jobs=-1)
       
    else:
        model=svm.SVC(C=1.0)

            
        
    score_means = list()
    score_stds = list()
    thresholds = (5, 10, 25 , 50 , 75, 100, 150 , 200, 500, 750, 1000, 1500, features.shape[1]-1)

    for thresh in thresholds:
         featuresfin=features[:,indices[:thresh]]
     # Compute cross-validation score using 1 CPU
         this_scores = cross_val_score(model, featuresfin, labels, cv=5,n_jobs=1)
         score_means.append(this_scores.mean())
         score_stds.append(this_scores.std())

    plt.errorbar(thresholds, score_means, np.array(score_stds))

    plt.title(
               'Performance of the simple classifier/regressor varying the percentile of features selected')
    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')
    plt
    plt.axis('tight')
    plt.show()

    print(max(score_means),score_means.index(max(score_means)),np.where(score_means==np.max(score_means)))             
    return thresholds[score_means.index(max(score_means))]


def run_kfold_svm(kf,features,labels):
    # get k fold performance of linear svm 
    fold=0;meansvmscore=0
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        svmscore = lin_clf.score(X_test, y_test)
        meansvmscore += svmscore
        fold+=1
    return meansvmscore/fold

def run_kfold_ridgeregression(kf,features,labels,opt_alpha):
    # get kfold performance of ridge regression
    fold=0;meanridgescore=0
    for train_index, test_index in kf.split(features):  
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        reg=linear_model.Ridge(alpha=opt_alpha)
        reg.fit(X_train,y_train)
        r2_test = reg.score(X_test,y_test)
        meanridgescore+=r2_test
        fold+=1
        
    return meanridgescore/fold
        
def optimise_random_forest(numfolds,features,features2,labels,method,outputpath,rand,alpha=0.1):
    
    kf = KFold(n_splits=numfolds,shuffle=True,random_state=42)
    kf.get_n_splits(features)
    if method=='classification':
        base_comparison=run_kfold_svm(kf,features2,labels)
    elif method=='regression':
        base_comparison=run_kfold_ridgeregression(kf,features2,labels,alpha)
        
    print('base method performs as %f on this k fold split' %(base_comparison))
    best_score=-1
    opt_depth=5
    opt_f=1
    target=[];
    target.append([0 ,0 ,float(base_comparison)])
    #targetd=[];
  #  targetd.append([0 ,float(base_comparison)])
  #  targetf=[];
  #  targetf.append([0 ,float(base_comparison)])
    ind=0;
#parameter optimisation find optimimum number of tree estimators, depth of trees and number of variables tested per branch
    for num_est in [1000]:#, 50000]: # 10000, ,100000,1000000
        for depth in [ 2,3, 5, 7, 10,15, 25]: #5,7, 5,7, 9, 11, 15, 203, 7, 5, 5, 2, 5, 8, 10, 25, 50, 100]:
            for max_f in [1, 5,7,9 ,11,15, 20,50,75,100]:# [1, 10,15, 25, 50, 75, 100, 250, int(np.sqrt(features.shape[1])) ,int(features.shape[1]/10),5000,10000]: # 5, 10, 25, 50, 100, 500, np.sqrt(features.shape[1])]:
               # regression[1, 5, 10, 20,50,75,100]: #, 
                print('params',num_est,depth,max_f,features.shape[1])
                if max_f <= features.shape[1]:
                    fold=0; meanscore=0;
                    ####################### CROSSVALIDATE RANDOM FOREST ################
                    for train_index, test_index in kf.split(features):
                        X_train, X_test = features[train_index], features[test_index]
                        y_train, y_test = labels[train_index], labels[test_index]
                        if method=='classification':
                            print('classification')
                            rf = RandomForestClassifier(max_depth=depth, n_estimators=num_est, max_features=max_f, random_state=rand,n_jobs=8,oob_score=True)
                        elif method=='regression':
                            print('regression')
                            rf = RandomForestRegressor(n_estimators=num_est,max_features=max_f,max_depth=depth, random_state=rand,n_jobs=-1,oob_score=True)

                        
                        rf.fit(X_train, y_train)
                        pred = rf.predict(X_test)
                        score = rf.score(X_test, y_test)
                        scoreall = rf.score(features, labels)
                        # print( pd.crosstab(index=y_test, columns=pred, rownames=['actual'], colnames=['preds']))
                        print('fold: %d rf test score %f rf train score %f  max depth %d num est %d max f %f oob %f' % (fold,score, scoreall, depth, num_est,max_f, rf.oob_score_))
                        # paramter optimisation
                        
                        meanscore += score
                        fold +=1   
        
        
                    print('mean rf score is %f max depth %d max f %d' % (meanscore / (numfolds),  depth, max_f,))
                    target.append([depth,max_f,float(meanscore / (numfolds))])
                    #if max_f==10:
                   #      targetd.append([depth,float(meanscore / (numfolds))])
                         
                   # if depth==9:
                   #      targetf.append([max_f,float(meanscore / (numfolds))])
                         
                    if meanscore>best_score:
                        opt_depth=depth
                        opt_f=max_f
                        best_score=score
    print(target)
    np.savetxt(outputpath,np.asarray(target),fmt='%f')
   # np.savetxt(outputpath+'depth',np.asarray(targetd),fmt='%f')
  #  np.savetxt(outputpath+'features',np.asarray(targetf),fmt='%f')
#
    return opt_depth,opt_f