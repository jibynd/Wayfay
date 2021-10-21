import numpy as np
import re

def mse(preds, truth):
    return round(((preds - truth)**2).mean(), ndigits=5) 

def mae(preds, truth):
    return round(abs(preds - truth).mean() , ndigits=5) 

def get_name(model):
    rep = model.__str__()
    return re.findall(".+(?=.*\(.+)", rep)[0]

def to_latex(df):

    df = df.apply(lambda r: " & ".join([str(n) for n in r]), axis = 1)
    print(" \\\ \n".join([str(n) for n in df]))

def norm(diff):
    return np.square(diff).mean()

def phi(u, M, v = None, noInv = False):
    if not noInv:
        M = np.linalg.inv(M)
    if not v:
        v = u
    #if M.shape == (1,1):
        #val = M * u @ v
    #print(u.shape, M.shape, v.shape)
    return np.dot(np.dot(u.T, M) ,v)