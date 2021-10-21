from dataclasses import dataclass 
import numpy as np
import numbers
#from models import BayesModel
global _nIter, _gamma
_Iter = 100
_gamma = None
class Variable:
    def __init__(self, name, **kwargs):
        self.name = name 

class HyperParam(Variable):
    def __init__(self,  **kwargs):
        pass


class RandomVariable(Variable):
    def __init__(self, name, val_init,  **kwargs):
        super().__init__(name, )
        self.curr = val_init
        self._build_storage(val_init)

    def store(self, step):
        if self.storage.ndim > 1:
            p = self.storage.shape[1] #; print(_gamma)
            self.storage[step, np.arange(p)[_gamma == 1]] = self.curr
        else:        
            self.storage[step] = self.curr 

    def _build_storage(self, val):
        global _nIter
        if isinstance(val, numbers.Number):
            self.storage = np.repeat(np.nan, _Iter, axis = 0)
            self.storage[0] = val # matrix = numpy.empty(shape=(2,5),dtype='object')
        else:
            self.storage = np.repeat(val.reshape(1,-1), _Iter, axis=0)
            self.storage[0] = val.squeeze()

    def get(self):
        return self.curr 


class InverseGamma(RandomVariable):
    def __init__(self, name, val_init, shape, scale, size = 1):
        super().__init__(name, val_init, )
        self.shape = shape
        self.scale = scale
        self.size = size

    def generate(self, step):
        self.curr = 1/np.random.beta(self.shape, self.scale, size = self.size)
        self.store(step)

        #return super().generate(step)

class MultiNormal(RandomVariable):
    def __init__(self, name, val_init, mean, cov, size = 1):
        super().__init__(name, val_init, )
        self.m = mean
        self.C = cov
        self.size = size

    def generate(self, step):
        self.curr = np.random.multivariate_normal(self.m.squeeze(), self.C, size = self.size, check_valid = "warn").squeeze()
        #self.store(step)


