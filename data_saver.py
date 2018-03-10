import os
import sys
import nibabel as nib
import numpy as np
from PIL import Image
from itertools import product
import math
import glob
import csv
import pandas as pd
import time
start_time = time.time()

filenames = os.listdir("/Users/daria/Documents/Group diss/Group Project Data/Data")

# Loads all the files and saves the names in a arrays corresponding to each other
glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data"))

# scan_id = label_names[1][9:20]
# session_code = label_names[1][25:31]
# print scan_id, session_code

# Load data
path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
T1 = pd.read_csv(path+'new_T1_300.csv', header=None)
T2 = pd.read_csv(path+'new_T2_300.csv', header=None)
Volume = pd.read_csv(path+'new_Volume_300.csv', header=None)
pat_code = pd.read_csv(path+'PatCode.csv', header=None)
sub_code = pd.read_csv(path+'SessID.csv', header=None)

T1 = list(T1.values.flatten())
T1 = [100000*x for x in T1]

T2 = list(T2.values.flatten())
T2 = [100000*x for x in T2]

Vol = list(Volume.values.flatten())
pat_code = list(pat_code[pat_code.columns[1]].values.flatten())
sub_code = list(sub_code[sub_code.columns[1]].values.flatten())


#for p in range(1,len(sub_code)+1):
p = 1
# print label_names[i], sep
# loads corresponding data one sequentially
label_file = os.path.join('sub-' + pat_code[p] + '_ses-' + str(sub_code[p]) + '_drawem_all_labels.nii.gz')
# T1w_restore_brain_file = os.path.join(T1w_names[i])
# T2w_restore_brain_file = os.path.join( T2w_names[i])
print ' '
print 'Calculating for patient data: ', p
print 'sub_code : ', sub_code[p]

# load all feature data and label file
tissue_labels = nib.load(label_file)
tissue_data = tissue_labels.get_data()
shape = tissue_labels.get_shape()
affine = tissue_labels.affine
print ' '
print shape
# save feature maps as separate files
labels = np.array(tissue_data)
small = labels.min()
large = labels.max()
print ' '
print small
print large

print len(T1)
print T2[80:]
raw_input('Press <ENTER> to continue')
# for region in (range(int(small)+1, int(large)+1)) :
#     # loop over regions for label images
#     # save data to each region
#     t1_region = T1[labels == region]


# for l in range(len(T1)):
#     t1_region = T1[labels == region]

t1_all = np.zeros((shape[0],shape[1],shape[2]), dtype=np.float32)
t2_all = np.zeros((shape[0],shape[1],shape[2]), dtype=np.float32)
vol_all = np.zeros((shape[0],shape[1],shape[2]), dtype=np.float32)
for i, j, k in product(xrange(shape[0]), xrange(shape[1]), xrange(shape[2])):

    # Find mask value aka. region type 1-87
    val = int(tissue_data[i][j][k])
    if val == 0 :
        t1_all[i][j][k] = 0
        t2_all[i][j][k] = 0
        vol_all[i][j][k] = 0
        # print 'val 0', t1_all[i][j][k]
    elif val == 84 :
        t1_all[i][j][k] = 0
        t2_all[i][j][k] = 0
        vol_all[i][j][k] = 0
    elif val >= 85 :
        t1_all[i][j][k] = T1[val-2]
        t2_all[i][j][k] = T2[val-2]
        vol_all[i][j][k] = Vol[val-2]
        # print val
        # print T2[val-2]
        # raw_input('Press <ENTER> to continue')
    else :
        # save t1 data to region type, coords
        t1_all[i][j][k] = T1[val-1]
        t2_all[i][j][k] = T2[val-1]
        vol_all[i][j][k] = Vol[val-1]
        # print val
        # print t1_all[i][j][k]
        # print T1[val-1]
        # print T1
    #raw_input('Press <ENTER> to continue')

print t1_all.shape
print t1_all.max()
print 'saving'

t1_img = nib.Nifti1Image(t1_all, affine)
t1_filesave = str(pat_code[p])+'_'+str(sub_code[p])+'_t1_features.nii'
t2_img = nib.Nifti1Image(t2_all, affine)
t2_filesave = str(pat_code[p])+'_'+str(sub_code[p])+'_t2_features.nii'
vol_img = nib.Nifti1Image(vol_all, affine)
vol_filesave = str(pat_code[p])+'_'+str(sub_code[p])+'_vol_features.nii'

nib.save(t1_img, t1_filesave)
nib.save(t2_img, t2_filesave)
nib.save(vol_img, vol_filesave)
print("--- %s seconds ---" % (time.time() - start_time))
# https://github.com/MIRTK/DrawEM/blob/master/label_names/all_labels.csv
