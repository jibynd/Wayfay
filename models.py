import numpy as np
from variables import * #HyperParam, InverseGamma, Data
from utils import *
from copy import deepcopy


class BayesModel:
    def __init__(self, **kwargs):
        self.ld0 = 1 #HyperParam(1)
        self.nIter = 100
        self.curr_gamma = None
        self.variables = {}
        self.meta = {}
        self.data = {}

    def fit(self, y, X, **data): 
        self.init_data(y, X)
        self.def_globals()
        p = self.meta["p"]
        mu = Mu(np.zeros(p))
        mu.eval(0)
        B = Beta(mu)
        sig = Sigma(1)

        self.def_globals() 

        global _vars, _meta, _data, _gamma
        #_vars = self.variables
        #_meta = self.meta
        #_data = self.data
        #_gamma = self.curr_gamma
        
        keys = range(p)
        for i in range(1,self.nIter):
            # dashboard(i, by)
            sig.generate(i)
            lp = self.loglike()
            #print(lp)
            for k in keys:
                gholder = deepcopy(_gamma)
                _gamma[k] = abs(1 - _gamma[k])
                if sum(_gamma) == 0:
                    _gamma = gholder
                    next
                lp_new = self.loglike()
                gap = lp_new - lp
                if gap < np.log(np.random.uniform()):
                    _gamma = gholder
            self.curr_gamma = _gamma
            B.generate(i)


    def init_data(self, y, X):
        self.meta["n"], self.meta["p"] = X.shape
        self.curr_gamma = np.ones(self.meta["p"]) 
        self.data["y"] = Data("y", y.reshape(-1,1))
        self.data["X"] = Data("X", X, "right")
        self.data["X'X"] = Data("X'X", X.T @ X, "both") 
        self.data["Xy"] = Data("Xy", (X.T @ y).reshape(-1,1), "left") 
        self.data["yy"] = Data("yy", norm(y)) 


    def def_globals(self):
        global _vars, _meta, _data, _gamma, _nIter
        _vars = self.variables
        _meta = self.meta
        _data = self.data
        _gamma = self.curr_gamma
        _nIter = self.nIter

    def loglike(self):
        #X = self.data["X"].get()
        n = self.meta["n"]
        sig = self.variables["sig"].get() 
        #V = np.eye(p) * sig
        B = self.variables["B"]
        #Minv = B.M.get()
        #M_ = B.cov
        #m_ = B.mean
        #phi(B.mean, B.cov); self.data["yy"].get()/sig, phi(B.mu.get(), B.M.get(), noInv = True)
        #print("here", B.mean.shape, B.cov.shape)
        le = -0.5 * (-phi(B.mean, B.cov) + self.data["yy"].get()/sig + phi(B.mu.get(), B.M.get(), noInv = True)) 
        lp = -n/2 * sig + 0.5*(np.log(np.linalg.det(B.M.get())) + np.log(np.linalg.det(B.cov))) + le 
        return lp 



class Sigma(InverseGamma):
    def __init__(self, val_init, a0 = 1/100, b0 = 1/100, size = 1):
        super().__init__("sig", val_init, a0, b0, size)
        self.a0 = a0
        self.b0 = b0
        global _vars
        _vars[self.name] = self 

    def generate(self, step):
        global _vars, _meta, _data
        self.shape = self.a0 + _meta["n"]/2
        self.scale = self.b0 + norm(_data["y"].get() - np.dot(_data["X"].get() , _vars["B"].get()))  
        super().generate(step)

class Mu(RandomVariable):
    def __init__(self, val_init,):
        super().__init__("mu", val_init)

    def eval(self, step):
        self.get() 
        self.store(step)
    
    def get(self):
        global _data
        y, X, XtX = _data["y"].get(), _data["X"].get(), _data["X'X"].get()
        m = np.linalg.inv(XtX) @ X.T @ y
        self.curr = m.squeeze()
        return m 

class Beta(MultiNormal):
    def __init__(self, mu, ):
        self.mu = mu
        self.M  = BCov("M")
        #self._mean = None
        #self._cov = None
        super().__init__("B", mu.get(), mu.get(), self.M.get())
        global _vars
        _vars[self.name] = self 

    def generate(self, step):
        global _vars, _meta, _data
        #X = _data["X"].get()
        #p, n = X.shape[1], _meta["n"]
        #V = np.eye(n) * _vars["sig"].get() 
        #self.cov = np.linalg.inv(self.M.get() + phi(X, V))
        #print(self.cov.shape, self.M.get().shape, self.mu.get() .shape)
        #self.mean = self.cov @ (self.M.get() @ self.mu.get() + _data["Xy"].get() / _vars["sig"].get()) 
        self.m = self.mean.squeeze()
        #print(self.m.shape)
        self.C = self.cov 
        super().generate(step)

    @property
    def cov(self):
        #print("Cov stuff")
        global _vars, _meta, _data
        V = np.eye(_meta["n"]) * _vars["sig"].get() 
        return np.linalg.inv(self.M.get() + phi(_data["X"].get(), V))

    @property
    def mean(self):
        global _vars, _meta, _data
        #print("mean", self.cov.shape, self.M.get().shape, self.mu.get().shape, (np.dot(self.M.get(), self.mu.get()) + _data["Xy"].get() / _vars["sig"].get()).shape)
        return np.dot(self.cov,  (np.dot(self.M.get(), self.mu.get()) + _data["Xy"].get() / _vars["sig"].get()))

class Data:
    def __init__(self, name, value, side = None, dim = None): # side = left, right, both
        self.name = name
        self.value = value
        #self.dim = dim if dim else len(value.shape)
        self.side = side

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


class BCov(Data):
    def __init__(self, name):
        global _data
        super().__init__( name, _data["X'X"].value, side = "both") 

    def get(self):
        "# transform value"
        return super().get() 

