import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from Model import voSVM

def lin_kernel(X1, X2):
        return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=1):
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sqdist)

# Example Data:
use_kernel = rbf_kernel
n_classes = 5
n_dimensions = 2
n_samples = 2000
X, labels = make_blobs(n_samples=n_samples,
                       centers=n_classes,
                       n_features=n_dimensions,
                       random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=0)

# linear kernel
K = use_kernel(X_train, X_train)
model = voSVM(K, y_train)

svs = model.getSVIndices() # indices of support vectors

# measure training accuracy
Ktrain = use_kernel(X_test, X_train[svs])
plabels = model.predict(Ktrain)
right = np.sum(np.equal(y_test, plabels))

print('got ', right, ' of ', len(y_test), ' test accuracy = ', right/len(y_test))
