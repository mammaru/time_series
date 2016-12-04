#coding: utf-8
"""
State Space Model

Author: mammaru <mauma1989@gmail.com>

"""
import numpy as np
from pandas import DataFrame
from ..core.base import TimeSeriesModel
from ..util.matrix import *
from .kalman import KalmanMixin

def gen_data(self, model, N):
    sys_value = np.random.randn(model.sys_dim,1)
    obs_value = np.asarray(np.zeros((model.obs_dim,1)))
    i = 0
    while(i < N):
        sysi = np.matrix(sys_value)[:,i]
        sys_value = np.hstack((sys_value, model.sys_eq(sysi)))
        obs_value = np.hstack((obs_value, model.obs_eq(sysi)))
        i += 1
    sys_value = DataFrame(sys_value.T)
    obs_value = DataFrame(obs_value.T)
    return sys_value, obs_value #return as taple object


class DynamicLinearModel(TimeSeriesModel, KalmanMixin):
    """Dynamic Linear Model"""
    def __init__(self, observation_dimention, system_dimention, x0mean=None, x0var=None, F=None, Q=None, H=None, R=None):
        #self.obs_dim = observation_dimention
        #self.sys_dim = system_dimention
        #self.x0mean = np.matrix(np.random.randn(system_dimention, 1))
        #self.x0var = identity(system_dimention)#np.matrix(np.eye(k)) # fixed
        #self.F = np.matrix(np.random.randn(system_dimention,system_dimention))
        #self.F = np.matrix(DataFrame(self.F).applymap(lambda x: 0 if np.abs(x)>0.5 else x))
        #self.F[abs(self.F)<1] = 0 # make matrix sparse
        #self.Q = np.matrix(np.eye(system_dimention)) # system noise variance
        #self.H = np.matrix(np.eye(observation_dimention,system_dimention)) # observation matrix
        #self.R = np.matrix(np.diag(np.diag(np.random.rand(observation_dimention,observation_dimention)))) # observation noise variance
        super(DynamicLinearModel, self).__init__(obs_dim=observation_dimention,
                                              sys_dim=system_dimention,
                                              x0mean=x0mean or np.matrix(np.random.randn(system_dimention, 1)),
                                              x0var=x0var or identity(system_dimention), #np.matrix(np.eye(k)) # fixed
                                              F=F or np.matrix(np.random.randn(system_dimention,system_dimention)),
                                              Q=Q or np.matrix(np.eye(system_dimention)),
                                              H=H or np.matrix(np.eye(observation_dimention,system_dimention)),
                                              R=R or np.matrix(np.diag(np.diag(np.random.rand(observation_dimention,observation_dimention)))))
        self.name = 'Dynamic Linear Model'


    def __call__(self, data):
        return self.execute(data)

    def execute(self, data):
        assert 0, "Not implemented \"execute\" method"

#    def system_equation(self, x, **kwds):
#        F = kwds.pop("F", None) if "F" in kwds else self.F
#        Q = kwds.pop("Q", None) if "Q" in kwds else self.Q
#        return np.asarray(F * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.sys_dim]).tolist()[0], np.asarray(Q))).T)

#    def observation_equation(self, x, **kwds):
#        H = kwds.pop("H", None) if "H" in kwds else self.H
#        R = kwds.pop("R", None) if "R" in kwds else self.R
#        return np.asarray(self.H * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.obs_dim]).tolist()[0], np.asarray(self.R))).T)

