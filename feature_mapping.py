# Feature mapping

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
