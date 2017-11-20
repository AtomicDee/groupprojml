import os
import nibabel as nib
import numpy as np
from PIL import Image
from itertools import product
import math
import glob
import csv
import pandas as pd

filenames = os.listdir("Data")
# print filenames

# Loads all the files and saves the names in a arrays corresponding to each other
label_names = glob.glob(os.path.join("Data",'*_drawem_all_labels.nii.gz'))
T1w_names = glob.glob(os.path.join("Data",'*_T1w_restore_brain.nii.gz'))
T2w_names = glob.glob(os.path.join("Data",'*_T2w_restore_brain.nii.gz'))
# print 'Label files', label_names
# print 'T1W files', T1w_names
# print 'T2W files', T2w_names

# scan_id = label_names[1][9:20]
# session_code = label_names[1][25:31]
# print scan_id, session_code

# Load GA data
GA_all_data = pd.read_csv('GA.csv')
print 'all data : ', GA_all_data

# Setting a limit to the number of iterations based on the number of patients
lim = len(label_names)

# Sampling the patient codes and samples for data separation
pat_code = [0]*lim
sub_code = [0]*lim
for x in range(len(label_names)):
    pat_code[x] = str(label_names[x][25:31])
    sub_code[x] = str(label_names[x][9:20])

i = 0
step = 0
titles = ['scan ID','Birth Age','GA','Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']
df = []

while i < lim:
    # Splits the name of the current dataset through the '_'s.
    # This is to prevent data being misaligned for patients with multiple scans.
    sep = label_names[i].split("_")
    # print label_names[i], sep

    # loads corresponding data one sequentially
    Tissue_labels_file = os.path.join(label_names[i])
    T1w_restore_brain_file = os.path.join(sep[0] + '_' + sep[1] + '_T1w_restore_brain.nii.gz')
    T2w_restore_brain_file = os.path.join(sep[0] + '_' + sep[1] + '_T2w_restore_brain.nii.gz')
    # T1w_restore_brain_file = os.path.join(T1w_names[i])
    # T2w_restore_brain_file = os.path.join( T2w_names[i])
    print 'sub_code : ', sub_code[i]

    # load current data GA info
    GA_current = GA_all_data[GA_all_data['id'] == sub_code[i]]
    GA_current = GA_current.values.tolist()
    GA_length = len(GA_current)
    print 'GA_current', GA_current
    if GA_length == 1 :
        step = 0
        print 'step : ', step

    print ' '
    print 'Calculating for patient data: ', i+1

    # Load tissue label data
    # Tissue_labels_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_drawem_tissue_labels.nii.gz')
    tissue_labels = nib.load(Tissue_labels_file)
    # print tissue_labels

    # extract tissue label data
    tissue_data = tissue_labels.get_data()
    shape = tissue_labels.get_shape()
    affine = tissue_labels.affine
    # print tissue_labels.dataobj

    # Load T1 images and extract data
    # T1w_restore_brain_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_T1w_restore_brain.nii.gz')
    # If the dataset is missing, it will leave it blank but continue to loop over the data.
    T1w_cont = T2w_cont = 1
    try:
        T1w_im = nib.load(T1w_restore_brain_file)
    except:
        print 'T1w dataset was missing. Proceeding by leaving it blank.'
        T1w_cont = 0
    try:
        T2w_im = nib.load(T2w_restore_brain_file)
    except:
        print 'T2w dataset was missing. Proceeding by leaving it blank.'
        T2w_cont = 0

    if T1w_cont == 1:
        T1_units = T1w_im.header.get_xyzt_units() #mm, sec
        T1_dimensions = T1w_im.header.get_zooms() #0.5mm each way
        #convert dimensions to m
        T1_dimensions = np.divide(T1_dimensions, 100)
        # print T1w_im
        t1_data = np.array(T1w_im.get_data())
    else:
        T1_units = T1_dimensions = T1_data = '__Data__Missing__'

    # Load T2 images and extract data
    # T2w_restore_brain_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_T2w_restore_brain.nii.gz')
    # T2w_im = nib.load(T2w_restore_brain_file)
    if T2w_cont == 1:
        T2_units = T2w_im.header.get_xyzt_units() #mm, sec
        T2_dimensions = T2w_im.header.get_zooms() #0.5mm each way
        #convert dimensions to m
        T2_dimensions = np.divide(T2_dimensions, 100)
        # print T1w_im
        t2_data = np.array(T2w_im.get_data())
    else:
        T2_units = T2_dimensions = T2_data = '__Data__Missing__'

    # Single array of ROI data for 1 dataset
    """t1_all = np.zeros((9,shape[0],shape[1],shape[2]), dtype=np.float32)

    for i, j, k in product(xrange(shape[0]), xrange(shape[1]), xrange(shape[2])):

        # Find mask value aka. region type 1-9
        val = tissue_data[i][j][k]
        # save t1 data to region type, coords
        t1_all[int(val)-1][i][j][k] = t1_data[i][j][k]"""

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
            # Calculate volume, only needed once
            vol = len(t1_region)*reduce(lambda x, y: x*y, T1_dimensions)
        else:
            t1_region = t1_avg_intensity = '__Data__Missing__'
        # # Calculate volume, only needed once
        # vol = len(t1_region)*reduce(lambda x, y: x*y, T1_dimensions)

        # loop over regions for T2 images
        if T2w_cont == 1:
            t2_region = t2_data[labels == region]
            t2_region = t2_region[t2_region>0] # remove any 0 values
            t2_avg_intensity = np.mean(t2_region)
            # Calculate volume, only needed once
            vol = len(t2_region)*reduce(lambda x, y: x*y, T2_dimensions)
        else:
            t2_region = t2_avg_intensity = '__Data__Missing__'
        # Save all the data to a list
        reduced_data.append([GA_current[step][0],GA_current[step][1],GA_current[step][2],region,t1_avg_intensity, t2_avg_intensity, vol])
    # Check if subject has had more than one scan, if so, increment step in order
    # to append corresponding GA data for the further scans
    if GA_length > 1 :
        step += 1
        print 'step : ', step
    if (step+1) == GA_length :
        step = 0
        print 'step : ', step

    df.append(pd.DataFrame(reduced_data, columns = titles))


    # print 'Reduced Data : ', reduced_data

    i += 1

results = pd.concat(df, keys = pat_code)

results.to_csv('drawem_labels.csv')

print results
# https://github.com/MIRTK/DrawEM/blob/master/label_names/all_labels.csv
