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

# ROI extraction of T1 data to individual arrays
"""t1_r1 = np.zeros((shape), dtype=np.float32)
t1_r2 = np.zeros((shape), dtype=np.float32)
t1_r3 = np.zeros((shape), dtype=np.float32)
t1_r4 = np.zeros((shape), dtype=np.float32)
t1_r5 = np.zeros((shape), dtype=np.float32)
t1_r6 = np.zeros((shape), dtype=np.float32)
t1_r7 = np.zeros((shape), dtype=np.float32)
t1_r8 = np.zeros((shape), dtype=np.float32)
t1_r9 = np.zeros((shape), dtype=np.float32)"""

# Single array of ROI data for 1 dataset
t1_all = np.zeros((9,shape[0],shape[1],shape[2]), dtype=np.float32)


for i, j, k in product(xrange(shape[0]), xrange(shape[1]), xrange(shape[2])):

    # Find mask value aka. region type 1-9
    val = tissue_data[i][j][k]
    # save t1 data to region type, coords
    t1_all[int(val)-1][i][j][k] = t1_data[i][j][k]

    """if tissue_data[i][j][k] == 1 :
        t1_all[1][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 2 :
        t1_all[2][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 3 :
        t1_all[3][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 4 :
        t1_all[4][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 5 :
        t1_all[5][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 6 :
        t1_all[6][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 7 :
        t1_all[7][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 8 :
        t1_all[8][i][j][k] = t1_data[i][j][k]
    elif tissue_data[i][j][k] == 9 :
        t1_all[9][i][j][k] = t1_data[i][j][k]"""

t1_r1_img = nib.Nifti1Image(t1_all[0], affine)
nib.save(t1_r1_img, 't1_r1.nii')

t1_r2_img = nib.Nifti1Image(t1_all[1], affine)
nib.save(t1_r2_img, 't1_r2.nii')

t1_r3_img = nib.Nifti1Image(t1_all[2], affine)
nib.save(t1_r3_img, 't1_r3.nii')

t1_r4_img = nib.Nifti1Image(t1_all[3], affine)
nib.save(t1_r4_img, 't1_r4.nii')

t1_r5_img = nib.Nifti1Image(t1_all[4], affine)
nib.save(t1_r5_img, 't1_r5.nii')

t1_r6_img = nib.Nifti1Image(t1_all[5], affine)
nib.save(t1_r6_img, 't1_r6.nii')

t1_r7_img = nib.Nifti1Image(t1_all[6], affine)
nib.save(t1_r7_img, 't1_r7.nii')

t1_r8_img = nib.Nifti1Image(t1_all[7], affine)
nib.save(t1_r8_img, 't1_r8.nii')

t1_r9_img = nib.Nifti1Image(t1_all[8], affine)
nib.save(t1_r9_img, 't1_r9.nii')
