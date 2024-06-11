import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from Model import voSVM

def lin_kernel(X1, X2):
        return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=0.004):
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sqdist)

# Example Data:
use_kernel = rbf_kernel
#n_classes = 20
#n_dimensions = 2
#n_samples = 1000
#X, labels = make_blobs(n_samples=n_samples,
#                       centers=n_classes,
#                       n_features=n_dimensions,
#                       random_state=0)
X, labels = fetch_openml(data_id=28, as_frame=False, return_X_y=True, parser='liac-arff')
labels = labels.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=0)

# linear kernel
K = use_kernel(X_train, X_train)
model = voSVM(K, y_train)

svs = model.getSVIndices() # indices of support vectors

# measure training accuracy
Ktrain = use_kernel(X_test, X_train[svs])
plabels = model.predict(Ktrain)
right = np.sum(np.equal(y_test, plabels))

print('got ', right, ' of ', len(y_test), ' test accuracy = ', right/len(y_test))
#solving took 106.767 sec
#got  1110  of  1124  test accuracy =  0.9875444839857651

