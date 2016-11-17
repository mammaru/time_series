#coding: utf-8

import numpy as np
#import pandas as pd
from pandas import DataFrame, Series

def identity(p, k=None):
    if k is None:
        return np.matrix(np.eye(int(p)))
    elif k<=0:
        raise ValueError, "Second argument must be positive integer."
    else:
        return np.matrix(np.eye(int(p),int(k)))

def diag(arg):
    pass

def random(p, k):
    return np.matrix(np.random.randn(k,k))
