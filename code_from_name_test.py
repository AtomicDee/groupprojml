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

GA_all_data = pd.read_csv('unpickled_dHCP_demographics_filtered30-01-18.csv')
print 'all data : ', GA_all_data

# Setting a limit to the number of iterations based on the number of patients
lim = len(label_names)
# limit = 200;

# Sampling the patient codes and samples for data separation
pat_code = [0]*lim
sub_code = [0]*lim
for x in range(len(label_names)):
    sep = label_names[x].split("_")
    print 'sep : ', sep

    l = len(sep[0])
    pat_code[x] = sep[0][l-11:l]
    print 'pat code : ', pat_code[x]
    sub_code[x] = sep[1][4:]
    print 'sub code : ', sub_code[x]
    print ' ';


i = 0
step = 0
titles = ['scan ID','Gender','Birth Age','Scan Age','Region', 'T1 Average Intensity', 'T2 Average Intensity', 'Volume']
print titles;
df = []

while i<5:
    i+=1

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
