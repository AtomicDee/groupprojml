import os
import nibabel as nib
import numpy as np
from PIL import Image
from itertools import product

filenames = os.listdir("Data")
# print filenames

T1w_restore_brain_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_T1w_restore_brain.nii.gz')
T1w_im = nib.load(T1w_restore_brain_file)

Tissue_labels_file = os.path.join("Data", 'sub-CC00060XX03_ses-12501_drawem_tissue_labels.nii.gz')
tissue_labels = nib.load(Tissue_labels_file)

# print tissue_labels
tissue_data = tissue_labels.get_data()
shape = tissue_labels.get_shape()
affine = tissue_labels.affine
# print tissue_labels.dataobj

# print T1w_im
t1_data = T1w_im.get_data()

# Single array of ROI data for 1 dataset
t1_all = np.zeros((9,shape[0],shape[1],shape[2]), dtype=np.float32)


for i, j, k in product(xrange(shape[0]), xrange(shape[1]), xrange(shape[2])):

    # Find mask value aka. region type 1-9
    val = tissue_data[i][j][k]
    # save t1 data to region type, coords
    t1_all[int(val)-1][i][j][k] = t1_data[i][j][k]
