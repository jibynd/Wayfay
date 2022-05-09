from math import gamma
import numpy as np
from variables import * #HyperParam, InverseGamma, Data
from utils import *
from copy import deepcopy

class Selector(RandomVariable):
    def __init__(self, p):
        self.curr = np.ones(p)
        super().__init__("gams", self.curr)
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
                self.curr = self.holder
                result = next(self)
            else:
                result = self.curr
            return result
        else:
            raise StopIteration

    def backtrack(self):
        self.curr = self.holder

    def record(self, step):
        self.store(step, np.ones(self.p))


class GPrior:
    def __init__(self, v0 = 1, g = None, nIter = 100):
        self.v0 = v0
        #self.s0 = None # a function 
        self.g  = g 
        self.nIter = nIter
        self.curr_gamma = None
        self.variables = {}
        self.meta = {}
        self.data = {}

    def init_data(self, y, X):
        self.meta["n"], self.meta["p"] = X.shape
        if not self.g:
            self.g = self.meta["n"]
        self.meta["g"] = self.g
        self.curr_gamma = np.ones(self.meta["p"]) 
        self.data["y"] = Data("y", y.reshape(-1,1))
        self.data["X"] = Data("X", X, "right")
        self.data["X'X"] = Data("X'X", X.T @ X, "both") 
        self.data["Xy"] = Data("Xy", (X.T @ y).reshape(-1,1), "left") 
        self.data["yy"] = Data("yy", np.sum(y**2)) 

    def def_globals(self):
        global _vars, _meta, _data, _gamma, _nIter
        _vars = self.variables
        _meta = self.meta
        _data = self.data
        _gamma = self.curr_gamma
        _nIter = self.nIter

    def loglike(self):
        s0 = _vars["sig"].s0
        g, v0 = self.g, self.v0 
        X = self.data["X"].get(); n, p = X.shape 
        #SSR = self.data["yy"] - tdot(y.T, g/(1+g) * tdot(X, inv(XtX), X), y)
        SRRg = SRR(g, self.data["y"].get(), self.data["yy"].get(), X, self.data["X'X"].get())
        return -p/2*log(1+g) + v0/2*log(v0*s0) -(v0+n)/2*log(v0*s0 + SRRg)

    def fit(self, y, X, **data): 
        self.init_data(y, X)
        self.def_globals()

        p = self.meta["p"]; g = self.g
        s0 = 1 #Define s0 here
        sig = Sigma(self.v0, s0)
        B = Beta(p)
        gammas = Selector(p)

        self.def_globals() 

        global _vars, _meta, _data, _gamma

        for i in range(1, self.nIter):
            # dashboard(i, by)
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


   

class Zero(RandomVariable):
    def __init__(self, p,):
        super().__init__("mu", np.zeros(p)) 

    def eval(self, step):
        self.get() 
        self.store(step)
    
    def get(self):
        global _gamma
        m = np.zeros(np.sum(_gamma))
        self.curr = m.squeeze()
        return m.reshape(-1,1) 


class Beta(MultiNormal):
    def __init__(self, p, ):
        #self.mu = Zero(p)
        self.M  = "g*sigSq* inv(X'X)"
        super().__init__("B", np.zeros(p), self.priorCov)
        #global _vars
        _vars[self.name] = self 

    def generate(self, step):
        #global _vars, _meta, _data
        m = self.mean.squeeze() 
        if m.ndim == 0:
            m = [m]
        self.m = m
        self.C = self.cov 
        super().generate(step, _gamma)

    @property
    def priorCov(self):
        #global _vars, _meta, _data
        return _meta["g"] * _vars["sig"].get() * inv(_data["X'X"].get()) 

    @property
    def cov(self):
        #global _meta
        return self.priorCov / (1 + _meta["g"])

    @property
    def mean(self):
        #global _vars, _meta, _data
        g, y, X, XtX = _meta["g"], _data["y"].get(), _data["X"].get(), _data["X'X"].get()
        return g/(1+g) * ols(y, X, XtX)

class Sigma(InverseGamma):
    def __init__(self, v0, s0 = None):
        super().__init__("sig", 1, v0 / 2, v0*s0/2)
        self.v0 = v0
        #self.s0 = s0
        #global _vars
        _vars[self.name] = self 

    def generate(self, step):
        #global _vars, _meta, _data
        #print(self.rate)
        super().generate(step, _gamma)
    
    @property
    def s0(self):
        #global _data
        y, X, = _data["y"].get(), _data["X"].get()
        B0 = ols(y, X, _data["X'X"].get())
        return np.mean((y - dot(X, B0))**2)

    @property
    def shape(self):
        #global _meta
        self._shape = (self.v0 + _meta["n"]) / 2
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def rate(self):
        #global _vars, _meta, _data
        g, y, yy, X, XtX = _meta["g"], _data["y"].get(), _data["yy"].get(), _data["X"].get(), _data["X'X"].get()
        val = (self.v0*self.s0 + SRR(g, y, yy, X, XtX)) / 2
        return val.squeeze()

    @rate.setter
    def rate(self, value):
        self._rate = value 


def SRR(g, y, yy, X, XtX):
    return yy - g/(1+g) * tdot(y.T, X, ols(y, X, XtX)) 

def ols(y, X, XtX):
    return tdot(inv(XtX), X.T, y) 

class Data:
    def __init__(self, name, value, side = None, dim = None): # side = left, right, both
        self.name = name
        self.value = value
        self.side = side

    def get(self):
        if not self.side:
            val = self.value
        else:
            assert(self.side in ["left", "right", "both"])
            #global _gamma
            index = _gamma == 1
            if self.side == "left":
                val = self.value[index,:]
            elif self.side == "right":
                val = self.value[:,index]
            else:
                val = self.value[index, :][:, index]    
        return val
