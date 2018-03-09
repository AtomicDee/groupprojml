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

T1 = list(T1[T1.columns[1:]].values.flatten())
T2 = list(T2[T2.columns[1:]].values.flatten())
Vol = list(Volume[Volume.columns[1:]].values.flatten())
pat_code = list(pat_code[pat_code.columns[1:]].values.flatten())
sub_code = list(sub_code[sub_code.columns[1:]].values.flatten())


for i in range(1,len(sub_code)+1):

    # print label_names[i], sep

    # loads corresponding data one sequentially
    label_file = os.path.join('sub-' + pat_code[i] + '_ses-' + str(sub_code[i]) + '_drawem_all_labels.nii.gz')
    # T1w_restore_brain_file = os.path.join(T1w_names[i])
    # T2w_restore_brain_file = os.path.join( T2w_names[i])
    print ' '
    print 'Calculating for patient data: ', i
    print 'sub_code : ', sub_code[i]

    # load all feature data and label file
    tissue_labels = nib.load(label_file)
    t1_features = tissue_labels.get_data()
    t2_features = tissue_labels.get_data()
    volume_features = tissue_labels.get_data()
    shape = tissue_labels.get_shape()
    affine = tissue_labels.affine
    print ' '
    print shape
    print t1_features
    # save feature maps as separate files
    labels = np.array(tissue_data)
    small = labels.min()
    large = labels.max()
    print ' '
    print labels
    print small
    print large

    #for region in (range(int(small)+1, int(large)+1)) :
        # loop over regions for label images
        # save data to each region




# https://github.com/MIRTK/DrawEM/blob/master/label_names/all_labels.csv
