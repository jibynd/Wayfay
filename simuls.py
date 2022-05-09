from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import *
from copy import deepcopy

@dataclass
class SimData:
    betas: list = field(repr = False)
    cov_matrix: Any = field(repr = False)
    sig: float
    n: int
    mu: list = field(default = None, repr = False)
    p: int = None
    with_test: bool = False
    train_ratio: float = 0.5
    as_df = False
    save_memomy = False
    seed: int = None

    def __post_init__(self):
        if self.seed:
            np.random.seed(self.seed)
        if callable(self.cov_matrix):
            ls_p = range(self.p)
            self.cov_matrix  = [[self.cov_matrix(i,j) for i in ls_p] for j in ls_p]
        self.p = len(self.cov_matrix) #self.cov_matrix.shape[1]
        if not self.mu:
            self.mu = np.zeros(self.p)
        self.X = np.random.multivariate_normal(self.mu, self.cov_matrix, size = self.n, check_valid = "warn")
        self.y = np.dot(self.X, self.betas) + np.random.normal(scale = self.sig, size = self.n)
        if self.with_test:
            cut_off = int(self.train_ratio*self.n)
            for train in [True, False]:
                sim_object = deepcopy(self)
                sim_object.train_ratio = None
                sim_object.with_test = False
                sim_object.n = cut_off if train else self.n - cut_off
                sim_object.X = self.X[0:cut_off, :] if train else self.X[cut_off:, :]
                sim_object.y = self.y[0:cut_off] if train else self.y[cut_off:]
                if train:
                    self.train = sim_object
                    if self.as_df:
                        self.train.to_df()
                else:
                    self.test = sim_object
                    if self.as_df:
                        self.test.to_df()
        if not self.with_test and self.as_df:
            self.to_df()

    def resimul(self):
        self.__post_init__()
    def to_df(self):
        self.df = pd.DataFrame(self.X, columns = ["x" + str(n) for n in np.arange(self.p)]) 
        self.df["y"] = self.y
        if self.save_memomy:
            self.X = None
            self.y = None 


@dataclass
class ExampleThree:
    sig: float 
    n: int
    with_test: bool = False
    seed: int = None

    def __post_init__(self):
        B = np.zeros(100)
        B[np.arange(9,100,10)] = 5
        B[np.arange(9,50,10)] = 0.5
        def cov(i, j):
            return 0.5**(abs(i-j))
        self.data = SimData(betas=B, cov_matrix= cov, sig= self.sig, n = self.n, with_test=self.with_test, seed = self.seed, p = 100)

@dataclass
class ExampleTwo:
    sig: float 
    n: int
    with_test: bool = False
    seed: int = None

    def __post_init__(self):
        B = np.array([5.6,5.6,5.6,0])
        def cov(i, j):
            if i == j:
                val = 1
            elif (i < 3 or j < 3):
                val = -0.39
            elif i == 3 or j == 3:
                val = 0.23
            return val
        self.data = SimData(betas=B, cov_matrix= cov, sig= self.sig, n = self.n, with_test=self.with_test, seed = self.seed, p = 4)

@dataclass
class ExampleOne:
    sig: float 
    n: int
    with_test: bool = False
    seed: int = None

    def __post_init__(self):
        B = np.array([3, 1.5, 0.1, 0.1, 2, 0, 0, 0])
        def cov(i, j):
            return 0.5**(abs(i-j))
        self.data = SimData(betas=B, cov_matrix= cov, sig= self.sig, n = self.n, with_test=self.with_test, seed = self.seed, p = 8)



@dataclass
class ExampleSelectThree:
    sig: float 
    n: int
    with_test: bool = False
    seed: int = None

    def __post_init__(self):
        B = np.zeros(100)
        B[np.arange(9,100,10)] = 5
        #B[np.arange(9,50,10)] = 0.5
        def cov(i, j):
            return 0.5**(abs(i-j))
        self.data = SimData(betas=B, cov_matrix= cov, sig= self.sig, n = self.n, with_test=self.with_test, seed = self.seed, p = 100)

@dataclass
class ExampleSelectTwo:
    sig: float 
    n: int
    with_test: bool = False
    seed: int = None

    def __post_init__(self):
        B = np.array([5.6,5.6,5.6,0])
        def cov(i, j):
            if i == j:
                val = 1
            elif (i < 3 or j < 3):
                val = -0.39
            elif i == 3 or j == 3:
                val = 0.23
            return val
        self.data = SimData(betas=B, cov_matrix= cov, sig= self.sig, n = self.n, with_test=self.with_test, seed = self.seed, p = 4)

@dataclass
class ExampleSelectOne:
    sig: float 
    n: int
    with_test: bool = False
    seed: int = None

    def __post_init__(self):
        B = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
        def cov(i, j):
            return 0.5**(abs(i-j))
        self.data = SimData(betas=B, cov_matrix= cov, sig= self.sig, n = self.n, with_test=self.with_test, seed = self.seed, p = 8)