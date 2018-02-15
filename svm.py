from sklearn import svm
import cPickle
import gzip
def load(file_name):
    # load the model
    stream = gzip.open(file_name, "rb")
    model = cPickle.load(stream)
    stream.close()
    return model

#X = [n_sample, n_features]
X =
#Y = Class labels
Y =
estimator.get_params()

model = svm.svc(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)


def save(file_name, model):
    # save the model
    stream = gzip.open(file_name, "wb")
    cPickle.dump(model, stream)
    stream.close()
#To save
save("/tmp/features_matrix", features_matrix)
save("/tmp/labels", labels)
save("/tmp/test_feature_matrix", test_feature_matrix)
save("/tmp/test_labels", test_labels)
#To load
features_matrix = load("/tmp/features_matrix")
labels = load("/tmp/labels")
test_feature_matrix = load("/tmp/test_feature_matrix")
test_labels = load("/tmp/test_labels")
