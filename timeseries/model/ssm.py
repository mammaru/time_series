#coding: utf-8
import numpy as np
from pandas import DataFrame

class StateSpaceModel:
    def __init__(self, p, k):
        self.obs_dim = p
        self.sys_dim = k
        self.x0mean = np.matrix(np.random.randn(k, 1))
        self.x0var = np.matrix(np.eye(k)) # fixed
        self.F = np.matrix(np.random.randn(k,k)) # system transition matrix
        #self.F = np.matrix(DataFrame(self.F).applymap(lambda x: 0 if np.abs(x)>0.5 else x))
        self.F[abs(self.F)<1] = 0 # make matrix sparse
        self.Q = np.matrix(np.eye(k)) # system noise variance
        self.H = np.matrix(np.eye(p,k)) # observation transition matrix
        self.R = np.matrix(np.diag(np.diag(np.random.rand(p,p)))) # observation noise variance
    
    def sys_eq(self, x, **kwds):
        F = kwds.pop("F", None) if "F" in kwds else self.F
        Q = kwds.pop("Q", None) if "Q" in kwds else self.Q
        return np.asarray(F * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.sys_dim]).tolist()[0], np.asarray(Q))).T)
    
    def obs_eq(self, x, **kwds):
        H = kwds.pop("H", None) if "H" in kwds else self.H
        R = kwds.pop("R", None) if "R" in kwds else self.R
        return np.asarray(self.H * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.obs_dim]).tolist()[0], np.asarray(self.R))).T)
    
    def gen_data(self, N):
        sys_value = np.random.randn(self.sys_dim,1)
        obs_value = np.asarray(np.zeros((self.obs_dim,1)))
        i = 0
        while(i < N):
            sysi = np.matrix(sys_value)[:,i]
            sys_value = np.hstack((sys_value, self.sys_eq(sysi)))
            obs_value = np.hstack((obs_value, self.obs_eq(sysi)))
            i += 1
        sys_value = DataFrame(sys_value.T)
        obs_value = DataFrame(obs_value.T)
        return sys_value, obs_value #return as taple object
