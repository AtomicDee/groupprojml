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
label_names = glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data",'*_drawem_all_labels.nii.gz'))

# scan_id = label_names[1][9:20]
# session_code = label_names[1][25:31]
# print scan_id, session_code

# Load data
path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
T1 = pd.read_csv(path+'new_T1.csv')
T2 = pd.read_csv(path+'new_T2.csv')
Volume = pd.read_csv(path+'new_Volume.csv')
pat_code = pd.read_csv()
T1 = T1[T1.columns[1:]]
T2 = T2[T2.columns[1:]]
Vol = Volume[Volume.columns[1:]]
SA = ScanAge[ScanAge.columns[1]]
BA = BirthAge[BirthAge.columns[1]]


for i in len(range(sub_code)):

    # print label_names[i], sep

    # loads corresponding data one sequentially
    Tissue_labels_file = os.path.join(label_names[i])
    label_file = os.path.join('sub-' + pat_code + '_ses-' + sub_code + '_drawem_all_labels.nii.gz')
    # T1w_restore_brain_file = os.path.join(T1w_names[i])
    # T2w_restore_brain_file = os.path.join( T2w_names[i])
    print ' '
    print 'Calculating for patient data: ', i
    print 'sub_code : ', sub_code[i]

    # load all feature data and label file
    tissue_labels = nib.load(label_file)


    # split features into T1, T2, Volumes

    # save feature maps as separate files



    # extract tissue label data
    tissue_data = tissue_labels.get_data()
    shape = tissue_labels.get_shape()
    affine = tissue_labels.affine
    # print tissue_labels.dataobj
    i += 1

    if T1w_cont == 1:
        T1_units = T1w_im.header.get_xyzt_units() #mm, sec
        T1_dimensions = T1w_im.header.get_zooms() #0.5mm each way
        #convert dimensions to m
        T1_dimensions = np.divide(T1_dimensions, 100)
        # print T1w_im
        t1_data = np.array(T1w_im.get_data())


    labels = np.array(tissue_data)
    small = labels.min()
    large = labels.max()
    reduced_data = []

    for region in (range(int(small)+1, int(large)+1)) :
        # loop over regions for T1 images
        if T1w_cont == 1:
            t1_region = t1_data[labels == region]
            t1_region = t1_region[t1_region>0] # remove any 0 values
            t1_avg_intensity = np.mean(t1_region)



print results
# https://github.com/MIRTK/DrawEM/blob/master/label_names/all_labels.csv
