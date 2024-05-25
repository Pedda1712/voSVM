import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from Model import voSVM

# Example Data:
n_centers = 3
n_features = 2
n_samples = 400
X, labels = make_blobs(n_samples=n_samples,
                       centers=n_centers,
                       n_features=n_features,
                       random_state=0)

def dotSimX(x1, x2): # euclid distance
	return x1.T @ x2

model = voSVM(X, X @ X.T, labels, dotSimX)

xs = X[:, 0]
ys = X[:, 1]

# plot points and SVs
fig, ax = plt.subplots()
ax.scatter(xs, ys, c=labels)

svinds = model.getSVIndices()
for ind in svinds: # mark support vectors
	ax.scatter(xs[ind], ys[ind], c='r', marker='x')


# training accuracy
right = 0
for i in range(0, n_samples):
	myX = X[i]
	myL = labels[i]
	pred = model.decide(myX)
	if pred == myL:
	 	right = right + 1
print('got ', right, ' of ', n_samples, ' training accuracy = ', right/n_samples)


if n_features != 2:
	plt.show()
	exit()

# visualize Separation by testing points on a grid (if model is for 2D-Data)
maxX = max(xs)
minX = min(xs)
maxY = max(ys)
minY = min(ys)

fig, ax = plt.subplots()

nXs = []
nYs = []
nLs = []

res = 50
for mx in np.linspace(minX, maxX, res):
	for my in np.linspace(minY, maxY, res):
		pred = model.decide(np.array([mx, my]))
		nXs.append(mx)
		nYs.append(my)
		nLs.append(pred)
ax.scatter(nXs, nYs, c=nLs)
plt.show()
