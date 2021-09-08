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
    pass
