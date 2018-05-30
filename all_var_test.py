#Adaboost regression split term/preterm training
import pandas as pd
import nibabel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



path_term = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/split_data/Term_Data/"
Term_data = pd.read_csv(path_term+'T1_T2_Vol.csv')
BirthAge_term = pd.read_csv(path_term+'BirthGA.csv')
ScanAge_term = pd.read_csv(path_term+'ScanGA.csv')

t_data = Term_data[Term_data.columns[1:]]
SA_t = ScanAge_term[ScanAge_term.columns[1]]
BA_t = BirthAge_term[BirthAge_term.columns[1]]

path_preterm = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/split_data/preterm_abov_37/"
Preterm_data = pd.read_csv(path_preterm+'T1_T2_Vol.csv')
BirthAge_pre = pd.read_csv(path_preterm+'BirthGA.csv')
ScanAge_pre = pd.read_csv(path_preterm+'ScanGA.csv')

p_data = Preterm_data[Preterm_data.columns[1:]]
SA_p = ScanAge_pre[ScanAge_pre.columns[1]]
BA_p = BirthAge_pre[BirthAge_pre.columns[1]]

rng = np.random.RandomState(1)
training_data = pd.concat([t_data,SA_t], axis = 1)
training_labels = BA_t

data = []
df = []

raw_input('ENTER')

testing_data = pd.concat([p_data,SA_p], axis=1)
testing_labels = BA_p


fields = ['MSE val','MAE val','FRI val','MSE std','MAE std','FRI std','min_s_split','min_s_leaf','min_feat','max_depth','n_est','ts']

N_est = [1, 10, 50, 100, 300]
ts = [0.5, 0.7, 0.9]
max_d = [2, 4, 10]
min_ss = [2, 4, 10]
min_sl = [1, 2, 4]
max_feat = ['auto','sqrt', 'log2']
i = 0
j = 0



last_hiscore = 0;

best_features = []
last_cv = 0;
feature_set = ()
best_set = ()
best_train_test = []


worst_cv = 99999
worst_set = ()
worst_train_test = []


fig, ax = plt.subplots(5, 3)
plt.title('Test plot')
fig, axes = plt.subplots(5, 3)
for mss in min_ss :
    for msl in min_sl :
        for mf in max_feat :
            for md in max_d :
                for n in N_est :
                    for t in ts :
                        # DecisionTreeRegressor(
                        # criterion=mse, 'mae', 'friedman_mse'
                        # splitter=best,
                        # max_depth=None, 2,4,10
                        # min_samples_split=2,4,10,
                        # min_samples_leaf=1,2,4
                        # min_weight_fraction_leaf=0.0,
                        # max_features=None, 'auto','sqrt', 'log2', float
                        # random_state=42
                        # max_leaf_nodes=None,
                        # min_impurity_decrease=0.0, --- can increase?
                        # presort=False --- may set to True)
                        regr_mse = AdaBoostRegressor(DecisionTreeRegressor(max_depth=md, min_samples_split=mss, min_samples_leaf=msl, max_features=mf),n_estimators=n, random_state=42)
                        regr_mae = AdaBoostRegressor(DecisionTreeRegressor(max_depth=md, min_samples_split=mss, min_samples_leaf=msl, max_features=mf),
                                                  n_estimators=n, random_state=42)
                        regr_fri = AdaBoostRegressor(DecisionTreeRegressor(max_depth=md, min_samples_split=mss, min_samples_leaf=msl,max_features=mf),
                                                  n_estimators=n, random_state=42)
                        regr_mse.fit(training_data, training_labels)
                        regr_mae.fit(training_data, training_labels)
                        regr_fri.fit(training_data, training_labels)

                        # Predict
                        mse_y = regr_mse.predict(training_data)
                        mse_z = regr_mse.predict(testing_data)
                        mae_y = regr_mae.predict(training_data)
                        mae_z = regr_mae.predict(testing_data)
                        fri_y = regr_fri.predict(training_data)
                        fri_z = regr_fri.predict(testing_data)

                        mse_sy = regr_mse.score(training_data, training_labels)
                        mse_sz = regr_mse.score(testing_data, testing_labels)

                        mae_sy = regr_mae.score(training_data, training_labels)
                        mae_sz = regr_mae.score(testing_data, testing_labels)

                        fri_sy = regr_fri.score(training_data, training_labels)
                        fri_sz = regr_fri.score(testing_data, testing_labels)

                        training_scores_mse = mse_sy
                        training_scores_mae = mae_sy
                        training_scores_fri = fri_sy
                        testing_scores_mse = mse_sz
                        testing_scores_mae = mae_sz
                        testing_scores_fri = fri_sz

                        mse_cv_y = cross_val_score(regr_mse, training_data, training_labels)
                        mse_cv_z = cross_val_score(regr_mse, testing_data, testing_labels)
                        mae_cv_y = cross_val_score(regr_mae, training_data, training_labels)
                        mae_cv_z = cross_val_score(regr_mae, testing_data, testing_labels)
                        fri_cv_y = cross_val_score(regr_fri, training_data, training_labels)
                        fri_cv_z = cross_val_score(regr_fri, testing_data, testing_labels)

                        training_accuracy_mse = mse_cv_y.mean()
                        testing_accuracy_mse = mse_cv_z.mean()
                        train_std_mse = mse_cv_y.std()
                        test_std_mse = mse_cv_z.std()

                        training_accuracy_mae = mae_cv_y.mean()
                        testing_accuracy_mae = mae_cv_z.mean()
                        train_std_mae = mae_cv_y.std()
                        test_std_mae = mae_cv_z.std()

                        training_accuracy_fri = fri_cv_y.mean()
                        testing_accuracy_fri = fri_cv_z.mean()
                        train_std_fri = fri_cv_y.std()
                        test_std_fri = fri_cv_z.std()

                        loscore = max([testing_accuracy_mse,testing_accuracy_mae,testing_accuracy_fri])

                        if loscore > last_hiscore :
                            print 'all vals : ', testing_accuracy_mse, ' ', testing_accuracy_mae, ' ', testing_accuracy_fri
                            print 'all std : ', test_std_mse, ' ', test_std_mae, ' ', test_std_fri
                            last_hiscore = max([testing_accuracy_mse,testing_accuracy_mae,testing_accuracy_fri])
                            last_vars = [mss, msl, mf, md, n, t]
                            data.append([testing_accuracy_mse,testing_accuracy_mae,testing_accuracy_fri,test_std_mse,test_std_mae,test_std_fri,mss, msl, mf, md, n, t])
                        print 'last stop : ', mss, ' ', msl, ' ', mf, ' ', md, ' ', n, ' ',t

                        df.append(pd.DataFrame(data, columns = fields))
df.to_csv('score_track.csv')
