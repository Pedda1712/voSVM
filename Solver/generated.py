"""
Sample code automatically generated on 2024-06-12 07:58:29

by geno from www.geno-project.org

from input

parameters
  matrix K symmetric
  matrix Ky symmetric
  matrix Y
  scalar c
variables
  vector a
min
  0.5*a'*(K.*Ky)*a-sum(a)
st
  a >= 0
  a <= c
  Y*a == vector(0)


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)



class GenoNLP:
    def __init__(self, K, Ky, Y, c, np):
        self.np = np
        self.K = K
        self.Ky = Ky
        self.Y = Y
        self.c = c
        assert isinstance(K, self.np.ndarray)
        dim = K.shape
        assert len(dim) == 2
        self.K_rows = dim[0]
        self.K_cols = dim[1]
        assert isinstance(Ky, self.np.ndarray)
        dim = Ky.shape
        assert len(dim) == 2
        self.Ky_rows = dim[0]
        self.Ky_cols = dim[1]
        assert isinstance(Y, self.np.ndarray)
        dim = Y.shape
        assert len(dim) == 2
        self.Y_rows = dim[0]
        self.Y_cols = dim[1]
        if isinstance(c, self.np.ndarray):
            dim = c.shape
            assert dim == (1, )
            self.c = c[0]
        self.c_rows = 1
        self.c_cols = 1
        self.a_rows = self.Ky_cols
        self.a_cols = 1
        self.a_size = self.a_rows * self.a_cols
        # the following dim assertions need to hold for this problem
        assert self.K_rows == self.Ky_rows == self.a_rows == self.Ky_cols == self.K_cols

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.a_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.c, inf)] * self.a_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.aInit = self.np.zeros((self.a_rows, self.a_cols))
        return self.aInit.reshape(-1)

    def variables(self, _x):
        a = _x
        return a

    def fAndG(self, _x):
        a = self.variables(_x)
        T_0 = (self.K * self.Ky)
        t_1 = (T_0).dot(a)
        f_ = ((0.5 * (a).dot(t_1)) - self.np.sum(a))
        g_0 = (((0.5 * t_1) - self.np.ones(self.a_rows)) + (0.5 * (T_0.T).dot(a)))
        g_ = g_0
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        a = self.variables(_x)
        f = (self.Y).dot(a)
        return f

    def gradientEqConstraint000(self, _x):
        a = self.variables(_x)
        g_ = (self.Y)
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        a = self.variables(_x)
        gv_ = ((self.Y.T).dot(_v))
        return gv_

def solve(K, Ky, Y, c, np, max_iter):
    start = timer()
    NLP = GenoNLP(K, Ky, Y, c, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver options, they can be omitted.
    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter' : max_iter,
               'm' : 10,
               'ls' : 0,
               'verbose' : 5  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.1.0')
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jacprod' : NLP.jacProdEqConstraint000})
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options,
                      constraints=constraints, np=np)
    else:
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jac' : NLP.gradientEqConstraint000})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints)

    # assemble solution and map back to original problem
    a = NLP.variables(result.x)
    elapsed = timer() - start
    print('solving took %.3f sec' % elapsed)
    return result, a

def generateRandomData(np):
    np.random.seed(0)
    K = np.random.randn(3, 3)
    K = 0.5 * (K + K.T)  # make it symmetric
    Ky = np.random.randn(3, 3)
    Ky = 0.5 * (Ky + Ky.T)  # make it symmetric
    Y = np.random.randn(3, 3)
    c = np.random.rand(1)[0]
    return K, Ky, Y, c

if __name__ == '__main__':
    import numpy as np
    # import cupy as np  # uncomment this for GPU usage
    print('\ngenerating random instance')
    K, Ky, Y, c = generateRandomData(np=np)
    print('solving ...')
    result, a = solve(K, Ky, Y, c, np=np)

