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
raw_input('Press <ENTER> to continue')
print filenames
raw_input('Press <ENTER> to continue')
# Loads all the files and saves the names in a arrays corresponding to each other
label_names = glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data",'*_drawem_all_labels.nii.gz'))
T1w_names = glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data",'*_T1w_restore_brain.nii.gz'))
T2w_names = glob.glob(os.path.join("/Users/daria/Documents/Group diss/Group Project Data/Data",'*_T2w_restore_brain.nii.gz'))


os.rename(T1w_names, T1w_names[:12]);
