import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

path_full = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/reg_full_results/"
path_red = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/reg_red_results/"


# Reduced Data
red_all_BA = pd.read_csv(path_red+'best_test_red_BA100_0.7.csv', header=None)
red_all_SA = pd.read_csv(path_red+'best_test_red_SA300_0.5.csv', header=None)

split_red_BA = len(red_all_BA)/2
split_red_SA = len(red_all_SA)/2

red_all_SA_lab = red_all_SA[:split_red_SA]
red_all_SA_pred = red_all_SA[split_red_SA:]
print len(red_all_SA_lab), len(red_all_SA_pred)

red_all_BA_lab = red_all_BA[:split_red_BA]
red_all_BA_pred = red_all_BA[split_red_BA:]
print len(red_all_BA_lab), len(red_all_BA_pred)

# Full Data
full_all_BA = pd.read_csv(path_full+'best_test_full_BA50_0.7.csv', header=None)
full_all_SA = pd.read_csv(path_full+'best_test_full_SA100_0.7.csv', header=None)

split_full_BA = len(full_all_BA)/2
split_full_SA = len(full_all_SA)/2

full_all_SA_lab = full_all_SA[:split_full_SA]
full_all_SA_pred = full_all_SA[split_full_SA:]
print len(full_all_SA_lab), len(full_all_SA_pred)

full_all_BA_lab = full_all_BA[:split_full_BA]
full_all_BA_pred = full_all_BA[split_full_BA:]
print len(full_all_BA_lab), len(full_all_BA_pred)

# Plotting
fig, ax = plt.subplots(2, 2)

# Full Data First Column
ax[0,0].scatter(full_all_BA_lab, full_all_BA_pred, c="b", label="Prediction")
ax[0,0].plot([0, 50],[0, 50],'--k')
ax[0,0].set_title("Full Set : Birth Age")
ax[0,0].set_xlabel('Actual')
ax[0,0].set_ylabel('Predicted')
ax[0,0].set_xlim([20, 50])
ax[0,0].set_ylim([20, 50])

ax[0,1].scatter(full_all_SA_lab, full_all_SA_pred, c='b', label='Prediction')
ax[0,1].set_title("Full Set : Scan Age")
ax[0,1].plot([0, 50],[0, 50],'--k')
ax[0,1].set_xlabel('Actual')
ax[0,1].set_ylabel('Predicted')
ax[0,1].set_xlim([20, 50])
ax[0,1].set_ylim([20, 50])
# Reduced Data Second Column
ax[1,0].scatter(red_all_BA_lab, red_all_BA_pred, c='b', label='Prediction')
ax[1,0].set_title("Reduced Set : Birth Age" )
ax[1,0].plot([0, 50],[0, 50],'--k')
ax[1,0].set_xlabel('Actual')
ax[1,0].set_ylabel('Predicted')
ax[1,0].set_xlim([20, 50])
ax[1,0].set_ylim([20, 50])

ax[1,1].scatter(red_all_SA_lab, red_all_SA_pred, c='b', label='Prediction')
ax[1,1].set_title("Reduced Set : Scan Age" )
ax[1,1].plot([0, 50],[0, 50],'--k')
ax[1,1].set_xlabel('Actual')
ax[1,1].set_ylabel('Predicted')
ax[1,1].set_xlim([20, 50])
ax[1,1].set_ylim([20, 50])

plt.legend()
plt.show()
