import pandas as pd
import nibabel
import os
import numpy as np


path = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/"
Tvals = pd.read_csv(path+'Normt1t2.csv')
print Tvals.shape
print ' '
print Tvals

T1 = Tvals[Tvals.columns[1:88]]
T2 = Tvals[Tvals.columns[88:]]

print T1.shape
print T2.shape
