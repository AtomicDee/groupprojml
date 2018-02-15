# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:12:25 2017

@author: emmar
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:17:29 2017

@author: emmar
"""
import pandas as pd
import numpy as np
import os 
import rfcode.helpfunctions as hf
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn import linear_model

#import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectPercentile

import nibabel

#### group average data
groupdirname='/vol/medic01/users/ecr05/HCP_PARCELLATION/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/'
labelsL=nibabel.load(os.path.join(groupdirname,'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LRtest.L.label.gii'))
labelsR=nibabel.load(os.path.join(groupdirname,'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LRtest.R.label.gii'))


####### paths to training and test data #############
dirname='/vol/medic01/users/ecr05/for_ricardo/' 
subjectpath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGDATA/TRAININGandTESTandVALlist.txt'  # load/preprocess all data as one and then randomly select test subjects 
csvpath='unrestricted.csv' # behavioural spreadsheet
datapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTrainTestValData.npy'#'/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGDATA/funcandlabelset_valandtrain.txt' #os.path.join(dirname,'GLASS_PRO_360_PCORR.mat') # 'GLASS_PRO_360_Z.mat' #
savedatapath=''#/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTrainData'
#datapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGDATA/funcandlabelset_valtestandtrain.txt'
#savedatapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTestData'
#testpath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTestData.npy'
#testsubjectpath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/TESTINGlist'
variables=['PMAT24_A_CR']#, 'PMAT24_A_RTCR', 'PMAT24_A_SI' ] # fluid intelligence 'PMAT24_A_CR' (correct responses) also PMAT24_A_SI (skipped items) PMAT24_A_RTCR (reaction time)
# explanation of behvioural variables https://wiki.humanconnectome.org/display/PublicData/HCP+Data+Dictionary+Public-+500+Subject+Release

numfolds=10
runPCA=False
RANDOM_STATE =42
numfolds=10
readtype='HCP_avfeatures' #'.func.gii' #'.mat'#
optimise_feature_selection=False
opt_alpha=10.0
# explanation of behvioural variables https://wiki.humanconnectome.org/display/PublicData/HCP+Data+Dictionary+Public-+500+Subject+Release
optimise_rf=True
opt_perc=20
opt_depth=5

################### PRE-PROCESS DATA #############################################

print('reading data matrix')
DATA=hf.read_DATA(datapath,readtype)
subjects=np.asarray(hf.get_subjects(DATA,subjectpath,readtype))

np.random.seed(42) # select same test subjects each time



if savedatapath!="":
    np.save(savedatapath,DATA)



################### PRE-PROCESS LABELS #############################################

print('reading  behavioural data')
xl_file = pd.read_csv(os.path.join(dirname,csvpath))  #open spreadsheet


alllabels=np.zeros((len(subjects),len(variables)))
deletedrows_all = []
top100indices_all=[]
for v_index,variable in enumerate(variables):
     print(v_index,variable)
     deletedrows=[]
     labels = []  # SELECT labels for subset (in future could choose a few and loop over)
     for index, subj in enumerate(subjects):
        # print(index,subj)
        if (xl_file.loc[xl_file['Subject'] == int(subj), variable].shape != (0,)):
            # print(xl_file.loc[xl_file['Subject']==int(subj),variables[0]].item())
            alllabels[index,v_index]=xl_file.loc[xl_file['Subject'] == int(subj), variable].item()
            labels.append(alllabels[index,v_index])
            
        else:
            # delete data for which there are no labels
            print('delete rows', index, subjects[index])
            deletedrows.append(index)
     
     DATA_v=np.delete(DATA, deletedrows, 0)
     print('features_v shape',DATA_v.shape)
     labels=np.asarray(labels)
     deletedrows_all.append(deletedrows)
     
     
     testingsubjectids=np.unique(np.random.choice(DATA_v.shape[0],70))
     testingDATA=DATA_v[testingsubjectids,:] #hf.read_DATA(testpath,readtype)
     testinglabels=labels[testingsubjectids]
    # testsubjects=subjects[testingsubjectids]
     trainingDATA= np.delete(DATA_v,np.sort(testingsubjectids),axis=0)
     traininglabels=np.delete(labels,np.sort(testingsubjectids),axis=0)
     # save testids for future runs
     np.save('/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/testsubjectids',testingsubjectids)
     
    
     ################### PRE-PROCESS #############################################

     if runPCA==True:
         ######## RUN dimensionality reduction with PCA #####################
         print('run PCA')
         pca = decomposition.PCA() # keep all components
         pca.fit(trainingDATA)
         featuresfin = pca.transform(trainingDATA)
         TestFeaturesfin=pca.transform(testingDATA)
     else:
         print('perform univariate testing of features and threshold feeatures to the top percentile ')
         # crude feature selection to remove most noisy features
         kbest=SelectPercentile(score_func=f_regression,percentile=opt_perc)  # this should be optimised as done for threshold on random forest importance selection (see l148)
         featuresbest = kbest.fit_transform(trainingDATA,  traininglabels)
         f_mask=kbest.get_support() # mask of features used
         indices1=np.where(f_mask==True)
         testbest=testingDATA[:,np.where(f_mask==True)[0]]
        ################ run random forest based feature selection ##################
    
        # print('features length after t-test based reduction',featuresbest.shape[1],indices1[0].shape)
         print('perform random forest based feature selection')
         model=RandomForestRegressor(n_estimators=1000,random_state=RANDOM_STATE,n_jobs=-1)
         model.fit(featuresbest, traininglabels)
         importances=model.feature_importances_
         print('after kbest selection rf score',model.score(testbest,testinglabels),model.score(featuresbest,traininglabels))

         print('rf importances', importances)
         std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
         indices = np.argsort(importances)[::-1]
    
         print('select the threshold on most important featutes by cross-validation')
         threshold=50 # hf.optimise_feature_selection_stage2(indices,featuresbest,traininglabels,'regression',RANDOM_STATE)
     
         featuresfin=featuresbest[:,indices[:threshold]]
         TestFeaturesfin=testbest[:,indices[:threshold]]

         top100indices=indices1[0][indices[:threshold]]
         top100indices_all.append(top100indices[:threshold])
     opt_f=9
     opt_depth=15
     
     ##################### OPTIMISE RANDOM FOREST  for each variable ##########################################
     print('Optimise Random Forest parameters for max depth and max features per node')
     #if optimise_rf==True:
     #    opt_depth,opt_f=hf.optimise_random_forest(5,featuresfin,featuresfin,traininglabels,'regression',os.path.join(dirname+'/paramopt/','rf_regressionparams'+ variables[0] + '.txt'),RANDOM_STATE,opt_alpha)
     
     clf=linear_model.Ridge(alpha=10.0) 
     clf.fit(featuresfin,traininglabels)
     ridgescore = clf.score(TestFeaturesfin,testinglabels)
     model=RandomForestRegressor(max_depth=opt_depth, n_estimators=10000, max_features=opt_f,random_state=RANDOM_STATE,n_jobs=-1)
     model.fit(featuresfin, traininglabels)

     rfscore=model.score(TestFeaturesfin,testinglabels)
     print('rftestscore is:',rfscore)
     print('ridgeregressionscore',ridgescore)
     importances=model.feature_importances_
     std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
     indices_fin = np.argsort(importances)[::-1]
     top50indices=indices1[0][indices_fin[:50]]
     # map selected features back to image domain
     featuresLfunc,featuresRfunc,featureslist=hf.map_feature_importances_back_to_image_space_HCP(110,360,top100indices,labelsL, labelsR)
    
    
     
     np.save(os.path.join(dirname+'/paramopt/','rf_features-'+variable),top100indices)
     np.save(os.path.join(dirname+'/paramopt/','rf_features-reprojected'+variable),featureslist)
     np.savetxt(os.path.join(dirname+'/paramopt/','rf_surfacefeatures-fin'+ variable + 'L.txt'),featuresLfunc)
     np.savetxt(os.path.join(dirname+'/paramopt/', 'rf_surfacefeatures-fin'+ variable + 'R.txt'), featuresRfunc)
        
#==============================================================================
#       
# print('delete',deletedrows_all,np.unique(deletedrows_all))
# features = np.delete(features, np.unique(deletedrows_all), 0)
# alllabels = np.delete(alllabels, np.unique(deletedrows_all), 0)
# top100indices_allvariables=np.unique(np.asarray(top100indices_all))
# np.save(os.path.join(dirname+'/paramopt/','rf_features-alltop100'),top100indices_allvariables)
# np.savetxt(os.path.join(dirname+'/paramopt/','rf_labels-alllabels'),alllabels,delimiter=',')
# featuresfin=features[:,top100indices_allvariables]
# np.savetxt(os.path.join(dirname+'/paramopt/','rf_features-all'),featuresfin,delimiter=',')
# 
# # if more than one column variance normalise prior to summing 
# if alllabels.shape[1] > 1:
#     print(np.mean(alllabels,axis=0))
#     alllabels=(alllabels - np.mean(alllabels,axis=0)) / np.std(alllabels,axis=0)
# 
# labels=np.sum(alllabels,axis=1)
# 
# print(labels.shape)
# 
# if optimise_rf==True:
#         opt_f,opt_depth=hf.optimise_random_forest(5,featuresfin,featuresfin,labels,'regression',os.path.join(dirname+'/paramopt/','rf_regressionparams_all_variables.txt'),RANDOM_STATE,opt_alpha)
# 
#==============================================================================





