import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from Model import voSVM

def lin_kernel(X1, X2):
        return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=1):
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sqdist)

# Example Data:
use_kernel = rbf_kernel
n_classes = 4
n_dimensions = 2
n_samples = 200
X, labels = make_blobs(n_samples=n_samples,
                       centers=n_classes,
                       n_features=n_dimensions,
                       random_state=0)

# linear kernel
K = use_kernel(X, X)
model = voSVM(K, labels)

xs = X[:, 0]
ys = X[:, 1]

# plot points and SVs
fig, ax = plt.subplots()
ax.scatter(xs, ys, c=labels)

svs = model.getSVIndices()
ax.scatter(X[svs,0], X[svs,1], c="r", marker="x")

# training accuracy
Ktrain = use_kernel(X, X[svs])
plabels = model.predict(Ktrain)
right = np.sum(np.equal(labels, plabels))

print('got ', right, ' of ', n_samples, ' training accuracy = ', right/n_samples)


if n_dimensions != 2:
        plt.show()
        exit()

# visualize Separation by testing points on a grid (if model is for 2D-Data)
maxX = max(xs)
minX = min(xs)
maxY = max(ys)
minY = min(ys)

res = 50

vX, vY = np.meshgrid(np.linspace(minX, maxX, res), np.linspace(minY, maxY, res))
vP = np.stack((vX.flatten(), vY.flatten()), axis=1)

Ktest = use_kernel(vP, X[svs])
labels = model.predict(Ktest)

fig, ax = plt.subplots()
ax.scatter(vP[:,0], vP[:,1], c=labels)
plt.show()

# calculate similarity between test points and SVs
