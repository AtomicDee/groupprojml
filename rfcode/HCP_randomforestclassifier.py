# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:17:29 2017

@author: emmar
"""

import pandas as pd
import nibabel
import numpy as np
import os 
import rfcode.helpfunctions as hf
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.feature_selection import SelectPercentile

#from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
groupdirname='/vol/medic01/users/ecr05/HCP_PARCELLATION/Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k/'
labelsL=nibabel.load(os.path.join(groupdirname,'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LRtest.L.label.gii'))
labelsR=nibabel.load(os.path.join(groupdirname,'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LRtest.R.label.gii'))
####### paths to training and test data #############
dirname='/vol/medic01/users/ecr05/for_ricardo/' 
#subjectpath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGandVALlist.txt'#/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGDATA/TRAININGandTESTandVALlist.txt'  # load/preprocess all data as one and then randomly select test subjects
csvpath='unrestricted.csv' # behavioural spreadsheet
datapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGVALandTESTgifti.txt'#rfTrainData.npy' #os.path.join(dirname,'GLASS_PRO_360_PCORR.mat') # 'GLASS_PRO_360_Z.mat' #
savedatapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGVALandTESTfeatures.txt'#/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTrainData'
#datapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TRAININGDATA/funcandlabelset_valtestandtrain.txt'
#savedatapath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTestData'
#testpath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/rfTestData.npy'
#testsubjectpath='/vol/medic01/users/ecr05/HCP_PARCELLATION/TESTINGDATA/TESTINGlist'
variables=['Gender']
variable=variables[0]
runPCA=False
optimise_feature_selection=False
optimise_rf=False
opt_perc=10
opt_threshold=500
opt_depth=5
opt_f=1
opt_alpha=0
RANDOM_STATE =42
numfolds=5
readtype='.func.gii' #HCP_avfeatures' #'.mat'#
# explanation of behvioural variables https://wiki.humanconnectome.org/display/PublicData/HCP+Data+Dictionary+Public-+500+Subject+Release


################### PRE-PROCESS DATA #############################################

print('reading data matrix')
DATA=hf.read_DATA(datapath,readtype)
subjects=np.asarray(hf.get_subjects(DATA,readtype))
FEATURES=hf.get_DATA(DATA,readtype)
np.random.seed(42) # select same test subjects each time



if savedatapath!="":
    np.save(savedatapath,DATA)


################### PRE-PROCESS LABELS #############################################
print('reading  behavioural data')
xl_file = pd.read_csv(os.path.join(dirname,csvpath))  #open spreadsheet 
subjects=hf.get_subjects(DATA,subjectpath,readtype)

labelvalues=[] # SELECT labels for subset (in future could choose a few and loop over)
deletedrows=[]

for index, subj in enumerate(subjects):
    # print(index,subj)
    if (xl_file.loc[xl_file['Subject'] == int(subj), variable].shape != (0,)):
        # print(xl_file.loc[xl_file['Subject']==int(subj),variables[0]].item())
        
        labelvalues.append(xl_file.loc[xl_file['Subject'] == int(subj), variable].item())
        
    else:
        # delete data for which there are no labels
        print('delete rows', index, subjects[index])
        deletedrows.append(index)
 
DATA_v=np.delete(FEATURES, deletedrows, 0)
print('features_v shape',DATA_v.shape)
labelvalues=np.asarray(labelvalues)
 
uniquelabels=np.unique(labelvalues) # CONVERT labels to indices #######
labels=np.zeros((len(labelvalues)))

for index, val in enumerate(uniquelabels):
    labels[np.where(np.asarray(labelvalues)==val)]=index

 
testingsubjectids=np.unique(np.random.choice(DATA_v.shape[0],70))
testingDATA=DATA_v[0:50,:] #testingsubjectids,:] #hf.read_DATA(testpath,readtype)
testinglabels=labels[0:50] #[testingsubjectids]
# testsubjects=subjects[testingsubjectids]
trainingDATA=DATA_v[51:DATA_v.shape[0],:] # np.delete(DATA_v,np.sort(testingsubjectids),axis=0)
traininglabels=labels[51:DATA_v.shape[0]]#np.delete(labels,np.sort(testingsubjectids),axis=0)
    

if runPCA==True:
         ######## RUN dimensionality reduction with PCA #####################
         print('run PCA')
         pca = decomposition.PCA() # keep all components
         pca.fit(trainingDATA)
         featuresfin = pca.transform(trainingDATA)
         TestFeaturesfin=pca.transform(testinglabels)
else:
         # crude feature selection to remove most noisy features
         kbest=SelectPercentile(percentile=opt_perc)  
         featuresbest = kbest.fit_transform(trainingDATA,  traininglabels)
         f_mask=kbest.get_support() # mask of features used
         indices1=np.where(f_mask==True)
         testbest=testingDATA[:,np.where(f_mask==True)[0]]
                              ################ run random forest based feature selection ##################
    
         print('features length after t-test based reduction',featuresbest.shape[1],indices1[0].shape)
    
         model=RandomForestClassifier(n_estimators=1000,random_state=RANDOM_STATE,n_jobs=-1)
         model.fit(featuresbest, traininglabels)
         importances=model.feature_importances_
         print('after kbest selection rf score',model.score(testbest,testinglabels),model.score(featuresbest,traininglabels))

         print('rf importances', importances)
         std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
         indices = np.argsort(importances)[::-1]
    
         if optimise_feature_selection=True:
             threshold=hf.optimise_feature_selection_stage2(indices, featuresbest, traininglabels, 'classifier', RANDOM_STATE)
         else:
            threshold=opt_threshold

         featuresfin=featuresbest[:,indices[:threshold]]
         TestFeaturesfin=testbest[:,indices[:threshold]]

         top100indices=indices1[0][indices[:threshold]]
        
    

opt_depth=25
opt_d=100

if optimise_rf==True:
   opt_depth,opt_f=hf.optimise_random_forest(5,featuresfin,featuresfin,traininglabels,'classification',os.path.join(dirname+'/paramopt/','rf_classicationparams'),RANDOM_STATE)

    
lin_clf = svm.LinearSVC()
lin_clf.fit(featuresfin,traininglabels)
svmscore = lin_clf.score(TestFeaturesfin,testinglabels)

# map selected features back to image domain and get test performace
model=RandomForestClassifier(max_depth=opt_depth, n_estimators=1000, max_features=opt_f,random_state=RANDOM_STATE,n_jobs=-1)
model.fit(featuresfin,traininglabels)

rfscore=model.score(TestFeaturesfin,testinglabels)
print('rftestscore is:',rfscore)
importances=model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices_fin = np.argsort(importances)[::-1]
top50indices=indices1[0][indices_fin[:50]]
featuresLfunc,featuresRfunc,featureslist=hf.map_feature_importances_back_to_image_space_HCP(110,360,top50indices,labelsL, labelsR)

# now optimise random forest params and compare performance against svm
np.save(os.path.join(dirname+'/paramopt/','rf_features-'+variable),top100indices)
np.save(os.path.join(dirname+'/paramopt/','rf_features-reprojected'+variable),featureslist)
np.savetxt(os.path.join(dirname+'/paramopt/','rf_surfacefeatures-fin'+ variable + 'L.txt'),featuresLfunc)
np.savetxt(os.path.join(dirname+'/paramopt/', 'rf_surfacefeatures-fin'+ variable + 'R.txt'), featuresRfunc)

