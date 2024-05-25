import numpy as np
from Solver import solve

# hypertet vertices
def mkVecLabel(cl, len):
	vec = np.zeros((len,1))
	for i in range(0, len):
		if i == cl:
			vec[i] = np.sqrt((len-1)/(len))
		else:
			vec[i] = (-1)/np.sqrt(len * (len-1))
	return vec

class voSVM:
	# X : data
	# K : data similarity matrix
	# labels : classification
	# simX : kernel function
	def __init__(self, X, K, labels, simX):
		self.noLabels = np.max(labels)+1
		self.X = X
		self.labels = labels
		self.simX = simX

		# construct label vector matrix
		self.Y = None
		for l in np.nditer(self.labels):
			vec = mkVecLabel(l, self.noLabels)
			if self.Y is None:
				self.Y = vec
			else:
				self.Y = np.append(self.Y, vec, axis=1)
		# solve
		self.kernel = K
		self.result, self.a = solve(self.kernel, self.Y, 9, np)
		
		# get indices of SVs
		self.svinds = []
		self.n_samples = self.labels.shape[0]
		for ind in range(0, self.n_samples):
			if self.a[ind] > 0.0:
				self.svinds.append(ind)
		
		# calculate offset vector b
		self.b = None
		for a1 in self.svinds: # calculate for every sv and average
			yi = mkVecLabel(self.labels[a1], self.noLabels) # correct label vector for this SV
			cb = yi * -1
			for a2 in self.svinds:
				# this is how you would do it for a kernelized binary SVM, lets just hope that it works ...
				cb = cb + (self.a[a2] * mkVecLabel(labels[a2], self.noLabels) * self.kernel[a1, a2])
			if self.b is None:
				self.b = cb
			else:
				self.b = self.b + cb
		self.b = self.b / len(self.svinds)


	def getSVIndices(self):
		return self.svinds

	# sv_indices : List of indices of Support vectors in X
	# test : x to test
	def decide(self, test):
		currentClass = None
		currentVal = None
		for t in range(0, self.noLabels):
			yt = mkVecLabel(t, self.noLabels)
			v = -(yt.T @ self.b)
			for i in self.svinds:
				ai = self.a[i]
				Ky = yt.T @ self.Y[:, i]
				Kphi = self.simX(self.X[i], test)
				v = v + ai * Ky * Kphi
			if (currentClass is None) or (currentVal < v):
				currentClass = t
				currentVal = v
		return currentClass
