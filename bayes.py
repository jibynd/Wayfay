from math import gamma
import numpy as np
from variables import * #HyperParam, InverseGamma, Data
from utils import *
from copy import deepcopy
from scipy.stats import invwishart, wishart 
from scipy.linalg import sqrtm



class Wayfay:
    def __init__(self, ld0 = 1, d0 = 1, eps = 1/1000, nIter = 100):
        self.nIter = nIter
        self.curr_gamma = None
        self.variables = {}
        self.meta = {}
        self.data = {}
        self.meta["eps"] = eps
        self.meta["ld0"] = ld0
        self.meta["d0"] = d0

    def init_data(self, y, X):
        self.meta["n"], self.meta["p"] = X.shape
        self.curr_gamma = np.ones(self.meta["p"]) 
        self.data["y"] = Data("y", y.reshape(-1,1))
        self.data["X"] = Data("X", X, "right")
        self.data["X'X"] = Data("X'X", dot(X.T, X), "both") # X.T @ X
        self.data["Xy"] = Data("Xy", dot(X.T , y.reshape(-1,1)), "left") 
        self.data["yy"] = Data("yy", np.sum(y**2)) 
        self.data["KX"] = Data("KX", dot(X.T, X), "both") 
        self.data["KX"] = Data("KX", np.diag(np.diag(dot(X.T, X))) / self.meta["n"] , "both") 

    def def_globals(self):
        global _vars, _meta, _data, _gamma, _nIter
        _vars = self.variables
        _meta = self.meta
        _data = self.data
        _gamma = self.curr_gamma
        _nIter = self.nIter

    def loglike(self):
        n, p = self.data["X"].get().shape 
        ld = _vars["ld"].get()
        B = _vars["B"]
        ssr = _vars["sig"].rate
        return (log(p) - p*log(ld) + log(det(dot(B.priorCov, B.C_))) - n * log(ssr)) / 2

    def fit(self, y, X, **data): 
        nIter = self.nIter
        self.init_data(y, X)
        self.def_globals()

        mu = Mu(nIter) ; self.mu = mu
        sig = Sigma(nIter)
        ld = Lambda(self.meta["ld0"], nIter, fixed=True)
        W = Wishart(nIter, self.meta["eps"])
        B = Beta(nIter)
        
        gammas = Selector(self.meta["p"], nIter)

        self.def_globals() 
        global _vars, _meta, _data, _gamma

        for i in range(1, self.nIter):
            #dashboard(i, nIter)
            W.generate(i)
            ld.generate(i) 
            lp = self.loglike()
            for k in gammas:
                _gamma = k
                lp_new = self.loglike()
                gap = lp_new - lp
                if gap < log(np.random.uniform()):
                    gammas.backtrack()
                    _gamma = gammas.curr
            self.curr_gamma = _gamma
            sig.generate(i)
            B.generate(i)
            gammas.record(i)
        self.gammas = gammas 
        #print("Done")


def make_1d(m, twoD = False):
    m = m.squeeze() 
    if m.ndim == 0:
        if twoD:
            m = [[m]]
        else:
            m = [m]
    return m
   
 
class Wishart(RandomVariable):
    def __init__(self, nIter, eps = 1/1000):
        self.eps = eps
        super().__init__("W", self.Z[0], nIter) 
        self.generator = invwishart.rvs
        self.curr = self.Z # Just a hack
        self.Wstor = self.curr
        _vars[self.name] = self

    def get(self):
        index = _gamma == 1
        return self.Wstor[index, :][:, index]

    def generate(self, step):
        WIsq = self.generator(self.df, self.scale)
        self.curr = inv(sqrtm(make_1d(WIsq, twoD= True)))
        index = _gamma == 1
        self.Wstor[index, :][:, index] = self.curr
        #self.store(step, _gamma)

    @property
    def scale(self):
        B, ld, sig = _vars["B"].get(), _vars["ld"].get(), _vars["sig"].get() #; print(sig)
        mu, KX = _data["mu"].get(), _data["KX"].get() 
        diff = B - mu
        return make_psd(1/ld/sig * tdot(diff, diff.T, KX) + inv(self.Z))

    @property
    def df(self):
        d = _data["X'X"].get().shape[0] #GEt from Z size"
        return d + _meta["d0"] # _meta["p"] + _meta["d0"]

    @property
    def Z(self):
        mu = _data["mu"].get() 
        #return np.diag(make_1d(abs(mu).squeeze() + self.eps))**(1/1) # + np.eye(mu.size)
        return np.eye(mu.size) #/10

class Mu(RandomVariable):
    def __init__(self, nIter):
        super().__init__("mu", self.get(), nIter)
        _data[self.name] = self  

    def eval(self, step):
        self.get() 
        self.store(step)
    
    def get(self):
        y, X, XtX = _data["y"].get(), _data["X"].get(), _data["X'X"].get()
        self.curr = ols(y, X, XtX) * 0
        return self.curr 


class Beta(RandomVariable):
    def __init__(self, nIter):
        #priorC = _vars["ld"].get()*_vars["sig"].get()* inv(self.priorCov) 
        super().__init__("B", _data["mu"].get(), nIter)
        self.generator = np.random.multivariate_normal # check_valid = "warn")
        _vars[self.name] = self 

    def generate(self, step):
        self.curr = self.generator(make_1d(self.mean), self.cov) 
        self.store(step, _gamma) 

    @property
    def priorCov(self):
        W = _vars["W"].get()
        return tdot(W, _data["KX"].get(), W)  

    @property
    def mu_(self):
        return dot(self.priorCov, _data["mu"].get()) / _vars["ld"].get() + _data["Xy"].get()

    @property
    def C_(self):
        return inv(self.priorCov / _vars["ld"].get() + _data["X'X"].get()) 

    @property
    def cov(self):
        return _vars["sig"].get() * self.C_

    @property
    def mean(self):
        return dot(self.C_, self.mu_) 


class Sigma(RandomVariable):
    def __init__(self, nIter):
        #s0 = np.mean((_data["y"].get() - dot(_data["x"], _vars["B"].get()))**2)
        super().__init__("sig", self.s0, nIter)
        self.generator = np.random.gamma
        self.shape = _meta["n"] / 2
        _vars[self.name] = self 

    def generate(self, step):
        self.curr =  1 / self.generator(self.shape , 1/self.rate ) 
        self.store(step, _gamma)
    
    @property
    def s0(self):
        #global _data
        y, X, = _data["y"].get(), _data["X"].get()
        B0 = ols(y, X, _data["X'X"].get())
        return np.mean((y - dot(X, B0))**2) 

    @property
    def rate(self):
        #global _vars, _meta, _data
        yy, ld, mu = _data["yy"].get(), _vars["ld"].get(), _data["mu"].get()
        B = _vars["B"] 
        return (yy + phi(mu, B.priorCov, noInv=True)/ld - phi(B.mu_, B.C_, noInv=True)) / 2 

class Lambda(RandomVariable):
    def __init__(self, ld0, nIter, fixed = False):
        super().__init__("ld", ld0, nIter)
        self.generator = np.random.gamma
        self.shape = _meta["n"] / 2
        self.fixed = fixed 
        self.curr = ld0 
        _vars[self.name] = self 

    def generate(self, step):
        if not self.fixed:
            self.curr =  1 / self.generator(self.shape, 1/self.rate) 
        self.store(step, _gamma)

    @property
    def rate(self):
        mu = _data["mu"].get()
        B = _vars["B"] 
        return  phi(B.mean - mu, B.priorCov, noInv=True) / _vars["sig"].get() / 2


def SRR(g, y, yy, X, XtX):
    return yy - g/(1+g) * tdot(y.T, X, ols(y, X, XtX)) 

def ols(y, X, XtX):
    return tdot(inv(XtX), X.T, y) 

class Data:
    def __init__(self, name, value, side = None, dim = None): # side = left, right, both
        self.name = name
        self.value = value
        self.side = side

    def __getitem__(self, key):
        return self.get()

    def __call__(self):
        return self.get()

    def get(self):
        if not self.side:
            val = self.value
        else:
            assert(self.side in ["left", "right", "both"])
            global _gamma
            index = _gamma == 1
            if self.side == "left":
                val = self.value[index,:]
            elif self.side == "right":
                val = self.value[:,index]
            else:
                val = self.value[index, :][:, index]    
        return val

class Selector(RandomVariable):
    def __init__(self, p, nIter):
        self.curr = np.ones(p)
        super().__init__("gams", self.curr, nIter)
        self.p = p
        #self.holder = np.ones(p)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.p:
            k = self.i
            self.holder = deepcopy(self.curr)
            self.curr[k] = abs(1 - self.curr[k])
            self.i += 1
            if sum(self.curr) == 0:
                print("!!! BOOM !!!")
                self.curr = deepcopy(self.holder)
                result = self.holder #next(self)
                print(result)
            else:
                result = self.curr
            return result
        else:
            raise StopIteration

    def backtrack(self):
        self.curr = self.holder

    def record(self, step):
        self.store(step, np.ones(self.p))