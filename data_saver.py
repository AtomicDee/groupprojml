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
T1w_names = glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data",'*_T1w_restore.nii.gz'))
T2w_names = glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data",'*_T2w_restore.nii.gz'))

# scan_id = label_names[1][9:20]
# session_code = label_names[1][25:31]
# print scan_id, session_code

# Load GA data
GA_all_data = pd.read_csv('unpickled_dHCP_demographics_filtered30-01-18.csv')
# print 'all data : ', GA_all_data
# range_scans = 0
# Setting a limit to the number of iterations based on the number of patients
# lim = len(label_names[range_scans:range_scans+10])
lim = len(label_names)
print 'lim : ', lim;
print 'range : ', len(range(lim+1))
# limit = 200;

# Sampling the patient codes and samples for data separation
pat_code = [0]*int(lim)
print 'len pat code = ', len(pat_code)
sub_code = [0]*int(lim)
for x in range(0,lim-1):
    # sep = label_names[range_scans+x].split("_")
    sep = label_names[x].split("_")
    l = len(sep[0])
    pat_code[x] = sep[0][l-11:l]
    sub_code[x] = sep[1][4:]

i = 0
step = 0
titles = ['Pat ID','Session ID','Birth Age','Scan Age','Gender','Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']
df = []

while i < lim-1:

    # Splits the name of the current dataset through the '_'s.
    # This is to prevent data being misaligned for patients with multiple scans.
    sep = label_names[i].split("_")
    # print label_names[i], sep

    # loads corresponding data one sequentially
    Tissue_labels_file = os.path.join(label_names[i])
    T1w_restore_brain_file = os.path.join(sep[0] + '_' + sep[1] + '_T1w_restore.nii.gz')
    T2w_restore_brain_file = os.path.join(sep[0] + '_' + sep[1] + '_T2w_restore.nii.gz')
    # T1w_restore_brain_file = os.path.join(T1w_names[i])
    # T2w_restore_brain_file = os.path.join( T2w_names[i])
    print ' '
    print 'Calculating for patient data: ', i
    print 'sub_code : ', sub_code[i]

    # load current data GA info
    GA_current = GA_all_data[GA_all_data['id'] == pat_code[i]]
    GA_current = GA_current.values.tolist()
    GA_length = len(GA_current)
    print 'GA_current', GA_current
    if GA_length < 2 :
        step = 0
        print 'step : ', step
    if not GA_current :
        print "GA current is empty"
        continue

    # Load tissue label data
    # Tissue_labels_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_drawem_tissue_labels.nii.gz')
    tissue_labels = nib.load(Tissue_labels_file)
    # print tissue_labels

    # extract tissue label data
    tissue_data = tissue_labels.get_data()
    shape = tissue_labels.get_shape()
    affine = tissue_labels.affine
    # print tissue_labels.dataobj
    i += 1
    # Load T1 images and extract data
    # T1w_restore_brain_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_T1w_restore_brain.nii.gz')
    # If the dataset is missing, it will leave it blank but continue to loop over the data.
    T1w_cont = T2w_cont = 1
    try:
        T1w_im = nib.load(T1w_restore_brain_file)
    except:
        print 'T1w dataset was missing. Skipping patient.'
        T1w_cont = 0
        # continue
    try:
        T2w_im = nib.load(T2w_restore_brain_file)
    except:
        print 'T2w dataset was missing. Skipping Patient.'
        T2w_cont = 0
        # continue

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
