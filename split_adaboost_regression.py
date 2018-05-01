#Adaboost regression split term/preterm training
import pandas as pd
import nibabel
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


path_features = "/Users/daria/Documents/Group diss/Group Project Data/csv_data/feature_data/"

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



training_data = pd.concat([t_data,SA_t], axis = 1)
training_labels = BA_t

#raw_input('ENTER')

testing_data = pd.concat([p_data,SA_p], axis=1)
testing_labels = BA_p

N_est = [1, 10, 50, 100, 300]
ts = [0.5, 0.7, 0.9]
md = [2, 4, 8]
i = 0
j = 0

training_scores = np.zeros((5,1))
training_accuracy = np.zeros((5,1))
train_std = np.zeros((5,1))
testing_scores = np.zeros((5,1))
testing_accuracy = np.zeros((5,1))
test_std = np.zeros((5,1))


best_features = []
last_cv = 0;
feature_set = ()
best_set = ()
best_train_test = []

mf = math.sqrt(len(training_data))
print round(mf)
raw_input('ENTER')
worst_cv = 99999
worst_set = ()
worst_train_test = []
kernel = np.random.randint(0,200)
rnd = np.random.RandomState(42)
fig, ax = plt.subplots(5, 3)
plt.title('Test plot')
fig, axes = plt.subplots(5, 3)
for n in N_est :
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

    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_split=2, random_state=rnd),
                              n_estimators=n, learning_rate=0.1, random_state=rnd, loss='exponential')
    # r = DecisionTreeRegressor(random_state=rnd).get_params('random_state')
    print 'n feat: ', regr.n_features_
    raw_input('ENTER')
    regr.fit(training_data, training_labels)

    # Predict
    y = regr.predict(training_data)
    z = regr.predict(testing_data)
    sy = regr.score(training_data, training_labels)
    sz = regr.score(testing_data, testing_labels)

    training_scores[i] = sy
    testing_scores[i] = sz

    # print ' '
    # print 'Training scores --> n_est = %0.2f , ts = %0.2f : ' % (n, t), sy
    # print 'Testing scores --> n_est = %0.2f , ts = %0.2f: '% (n, t), sz
    # print ' '

    scores_z = cross_val_score(regr, testing_data, testing_labels)
    scores_y = cross_val_score(regr, training_data, training_labels)
    # print ' accuracy training: %0.2f (+/- %0.2f) ' % (scores_y.mean(), scores_y.std() *2)
    # print ' accuracy testing: %0.2f (+/- %0.2f) ' % (scores_z.mean(), scores_z.std() *2)

    training_accuracy[i][0] = scores_y.mean()
    testing_accuracy[i][0] = scores_z.mean()
    train_std[i][0] = scores_y.std()
    test_std[i][0] = scores_z.std()

    length_set = [len(training_labels)]
    # print 'length set: ', length_set
    # print 'length above: ', len(length_set)
    # print 'length train lab: ', len(training_labels)
    # print 'length y:', len(y)
    # print 'length test lab: ', len(testing_labels)
    # print 'length z: ', len(z)
    #
    # raw_input('ENTER')

    if (scores_z.mean() > 0.49000) and (scores_z.mean() > last_cv):
        best_features = regr.feature_importances_
        print len(best_features)
        last_cv = scores_z.mean()
        feature_set = (n)
        best_set = (n)
        best_train_test = np.concatenate((length_set, training_labels, y, testing_labels, z))
        print 'Best features updated at n=%d m=8' % (n)

    if scores_z.mean() < worst_cv :
        worst_cv = scores_z.mean()
        worst_set = (n)
        worst_train_test = np.concatenate((length_set, training_labels, y, testing_labels, z))

    # list_labels = np.concatenate(training_labels.values).ravel().tolist()

    # Plot the results

    print i
#     axes[i,j].scatter(training_labels, y, c="g", label="n_estimators=%0.2f" % n, linewidth=1)
#     axes[i,j].set_title("train n :%0.2f md:%0.2f" % (n, m))
#
#     ax[i,j].scatter(testing_labels, z, c="g", label="n_estimators=%0.2f" % n, linewidth=1)
#     ax[i,j].set_title("test n :%0.2f md:%0.2f" % (n, m))
#
    i+=1
    if i>4:
        i=0

# plt.show()
print 'best set'
print best_set
print 'training scores final'
print training_scores
print 'testing scores final'
print testing_scores
print 'training accuracy final'
print training_accuracy
print 'testing accuracy final'
print testing_accuracy
print 'training std'
print train_std
print 'testing std'
print test_std

if best_set :
    zeros = np.zeros((5,1));
    conc = np.concatenate((training_scores, zeros, testing_scores, zeros, training_accuracy, zeros, testing_accuracy, zeros, train_std, zeros, test_std), axis = 1)
    np.savetxt('all_scores_split_SA.csv', conc, fmt='%0.8f', delimiter=',')   # X is an array
    np.savetxt(path_features+'best_features_split_SA'+str(best_set[0])+'_'+str(8)+'.csv', best_features, fmt='%0.8f', delimiter=',')
    np.savetxt('best_scores_split_SA'+str(best_set[0])+'_'+str(8)+'.csv', best_train_test, fmt='%0.8f', delimiter=',')
    np.savetxt('worst_scores_split_SA'+str(worst_set[0])+'_'+str(8)+'.csv', worst_train_test, fmt='%0.8f', delimiter=',')
    print 'feature set'
    print feature_set, '\n'
    print 'Worst set'
    print worst_set, '\n'


# feature extraction

print len(best_features)


T1_feat = best_features[:86]
print len(T1_feat)
print type(T1_feat)
T2_feat = best_features[86:172]
print len(T2_feat)
Vol_feat = best_features[172:258]
print len(Vol_feat)
if len(best_features) == 258 :
    sa_feat = best_features[258]
#
# df1 = pd.DataFrame(T1_feat)
# df1.to_csv(path+"T1_split_SA.csv", header=None, index=None)
#
# df2 = pd.DataFrame(T2_feat)
# df2.to_csv(path+"T2_split_SA.csv", header=None, index=None)
#
# df3 = pd.DataFrame(Vol_feat)
# df3.to_csv(path+"Vol_split_SA.csv", header=None, index=None)
#
# df4 = pd.DataFram(sa_feat)
# df4.to_csv(path+"sa_split_SA.csv", header=None, index=None)
