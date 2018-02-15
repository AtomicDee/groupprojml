# RF implementation

# Features : GA Birth, GA Scan, Gender, Region, T1 intensity, T2 intensity, volume
# Max features = <sqrt(7) = 2
# Max depth : 3,5,7,9 - compare
# No_estimators : 10, 30, 100 - compare (generall the more the bettwer however the cost of
# learning increases and the benefit of learning decreases as you go up, 100 probably
# too many, maybe try 50 or 60 as well and compare)

sklearn.ensemble.RandomForestClassifier(
    n_estimators=10,
    criterion=’gini’,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=’auto’,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True, oob_score=False,
    n_jobs=1,
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None)

# RandomForestClassifier(X,Y) ---> X = data ; Y = labels
