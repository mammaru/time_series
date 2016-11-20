#coding: utf-8

import numpy as np
#import pandas as pd
#from pandas import DataFrame, Series

def identity(n, m=None):
    if n<=0:
        raise ValueError, "Second argument must be positive integer."
    elif m is None:
        m = n
    return np.matrix(np.eye(int(n),int(m)))

def diag(arg):
    pass

def random(p, k):
    return np.matrix(np.random.randn(k,k))

def normalize(mat):
    return (mat - mat.mean())/mat.std()
