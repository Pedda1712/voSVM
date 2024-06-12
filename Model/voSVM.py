import numpy as np
from Solver import solve


# hypertet vertices
def mkVecLabel(cl, len):
        vec = np.zeros((len, 1))
        for i in range(0, len):
                if i == cl:
                        vec[i] = np.sqrt((len-1)/(len))
                else:
                        vec[i] = (-1)/np.sqrt(len * (len-1))
        return vec

class voSVM:
        # K : data similarity matrix
        # labels : classification
        # max_iter : max inner iterations
        def __init__(self, K, labels, max_iter=200):
                self.noLabels = np.max(labels)+1
                self.labels = labels

                # construct label vector matrix
                self.Y = None
                for l in np.nditer(self.labels):
                        vec = mkVecLabel(l, self.noLabels)
                        if self.Y is None:
                                self.Y = vec
                        else:
                                self.Y = np.append(self.Y, vec, axis=1)
		# solve
                self.Ky = self.Y.T @ self.Y
                self.kernel = K
                print("start solving")
                self.result, self.a = solve(self.kernel, self.Ky, self.Y, 9, np, max_iter=max_iter)

                # get indices of SVs
                self.svinds = []
                self.n_samples = self.labels.shape[0]

                for ind in range(0, self.n_samples):
                        if self.a[ind] > 0.0:
                                self.svinds.append(ind)
                
                # these are the submatrices of Y, a, K belonging to the support vectors
                self.svY = self.Y[:, self.svinds]
                self.svA = np.repeat(self.a[self.svinds].reshape(1, -1), self.Y.shape[0], axis=0) # stacked alphas for element-wise mul
                self.svK = self.kernel[:, self.svinds][self.svinds, :]

                self.b = (-1 * self.svY) + (np.multiply(self.svA, self.svY) @ self.svK)
                # average the biases of all support vectors for robustness
                self.b = np.mean(self.b, axis=1)

        # return support vector indices
        def getSVIndices(self):
                return self.svinds
        
        # ktest: row index = test index, column index = sv index
        def predict(self, ktest):
                arow = self.svA[0, :].reshape(1, -1) # alphas as row vector
                scores = []
                for t in range(0, self.noLabels):
                        yt = mkVecLabel(t, self.noLabels)
                        ky = yt.T @ self.svY
                        sim = np.repeat(-(yt.T @ self.b), ktest.shape[0]).reshape(-1,1) + (ktest @ np.multiply(arow, ky).T)
                        scores.append(sim)
                
                scores = np.hstack(scores)
                return (np.argmax(scores, axis=1))

