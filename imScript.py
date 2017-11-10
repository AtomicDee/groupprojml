import os
import nibabel as nib
import numpy as np
from PIL import Image
from itertools import product
import math

filenames = os.listdir("Data")
# print filenames

# Load tissue label data
Tissue_labels_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_drawem_tissue_labels.nii.gz')
tissue_labels = nib.load(Tissue_labels_file)
# print tissue_labels

# extract tissue label data
tissue_data = tissue_labels.get_data()
shape = tissue_labels.get_shape()
affine = tissue_labels.affine
# print tissue_labels.dataobj

# Load T1 images and extract data
T1w_restore_brain_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_T1w_restore_brain.nii.gz')
T1w_im = nib.load(T1w_restore_brain_file)

T1_units = T1w_im.header.get_xyzt_units() #mm, sec
T1_dimensions = T1w_im.header.get_zooms() #0.5mm each way
#convert dimensions to m
T1_dimensions = np.divide(T1_dimensions, 100)
# print T1w_im
t1_data = np.array(T1w_im.get_data())

# Load T2 images and extract data
T2w_restore_brain_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_T2w_restore_brain.nii.gz')
T2w_im = nib.load(T2w_restore_brain_file)
T2_units = T2w_im.header.get_xyzt_units() #mm, sec
T2_dimensions = T2w_im.header.get_zooms() #0.5mm each way
#convert dimensions to m
T2_dimensions = np.divide(T2_dimensions, 100)
# print T1w_im
t2_data = np.array(T2w_im.get_data())

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
    t1_region = t1_data[labels == region]
    t1_region = t1_region[t1_region>0] # remove any 0 values
    t1_avg_intensity = np.mean(t1_region)
    # Calculate volume, only needed once
    vol = len(t1_region)*reduce(lambda x, y: x*y, T1_dimensions)

    # loop over regions for T2 images
    t2_region = t2_data[labels == region]
    t2_region = t2_region[t2_region>0] # remove any 0 values
    t2_avg_intensity = np.mean(t2_region)

    # Save all the data to a list
    reduced_data.append([region,t1_avg_intensity, t2_avg_intensity, vol])

print 'Reduced Data : ', reduced_data
