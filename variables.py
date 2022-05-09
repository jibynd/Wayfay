from dataclasses import dataclass 
import numpy as np
import numbers
from copy import deepcopy


class Variable:
    def __init__(self, name, **kwargs):
        self.name = name 

class HyperParam(Variable):
    def __init__(self,  **kwargs):
        pass


class RandomVariable(Variable):
    #nonlocal _Iter
    def __init__(self, name, val_init, nIter, **kwargs):
        super().__init__(name, )
        self.curr = val_init
        self._build_storage(val_init, nIter)

    def store(self, step, gamma):
        if self.storage.ndim > 1:
            p = self.storage.shape[1] #; print(_gamma)
            self.storage[step, np.arange(p)[gamma == 1]] = self.curr
        else:        
            self.storage[step] = self.curr 

    def _build_storage(self, val, nIter):
        if isinstance(val, numbers.Number):
            self.storage = np.repeat(np.nan, nIter, axis = 0)
            self.storage[0] = val # matrix = numpy.empty(shape=(2,5),dtype='object')
        else:
            self.storage = np.repeat(val.reshape(1,-1), nIter, axis=0)
            self.storage[0] = val.squeeze()

    def get(self):
        return self.curr 
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
            else:
                result = self.curr
            return result
        else:
            raise StopIteration

    def backtrack(self):
        self.curr = self.holder

    def record(self, step):
        self.store(step, np.ones(self.p))

class InverseGamma(RandomVariable):
    def __init__(self, name, shape, rate, nIter, val_init = None, size = 1):
        if not val_init:
            val_init = rate / shape
        super().__init__(name, val_init, nIter)
        self.shape = shape
        self.rate = rate
        self.size = size

    def generate(self, step, gamma):
        self.curr = 1/np.random.gamma(self.shape, 1/self.rate, size = self.size)
        self.store(step, gamma)

        #return super().generate(step)

class MultiNormal(RandomVariable):
    def __init__(self, name, mean, cov, nIter, val_init = None, size = 1):
        if not val_init:
            val_init = mean 
        super().__init__(name, val_init, nIter) 
        self.m = mean
        self.C = cov
        self.size = size

    def generate(self, step, gamma):
        self.curr = np.random.multivariate_normal(self.m, self.C, size = self.size, check_valid = "warn").squeeze()
        self.store(step, gamma)


