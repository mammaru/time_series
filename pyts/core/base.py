#coding: utf-8
import numpy as np
#import pandas as pd
from pandas import DataFrame

# TODO: map or apply is better
def acv(X, k):
    mu = X.mean() # Series
    y = X.values
    N, M = X.shape
    result = []
    for m in range(M):
        tmp = 0.0
        for n in range(k+1, N):
            tmp += (y[n,m] - mu[m])*(y[n-k,m] - mu[m])
        result.append(tmp/N)
    return np.array(result)
        
def acf(X, k):
    auto_cov = acv(X, k)
    auto_cov_0 = acv(X, 0)
    return auto_cov/auto_cov_0
        
# TODO: map or apply is better
def ccv(X, k):
    mu = X.mean() # Series
    y = X.values
    N, M = X.shape
    auto_cov = acv(X, k)
    result = np.diag(auto_cov)
    for l in range(M):
        for m in range(M):
            if l!=m:
                tmp = 0.0
                for n in range(k+1, N):
                    tmp += (y[n,l] - mu[l])*(y[n-k,m] - mu[m])
                result[l,m] = tmp/N
                result[m,l] = -tmp/N
    return result

# TODO: map or apply is better
def ccf(X, k):
    N, M = X.shape
    cross_cov = ccv(X, k)
    cross_cov_0 = ccv(X, 0)
    result = np.diag(acf(X, k))
    for i in range(M):
        for j in range(M):
            if i!=j:
                result[i,j] = cross_cov[i,j] / np.sqrt(cross_cov_0[i,i] * cross_cov_0[j,j])
    return result


class TimeSeriesModel(object):
    def __init__(self, **args):
        self.name = 'Time Series Model'
        self.__set_parameters(args.items())

    def __str__(self):
        return self.name

    def __set_parameters(self, parameters):
        self.__params = []
        for key, value in parameters:
            self.__params.append(key)
            setattr(self, key, value)

    @property
    def params(self):
        parameters = {}
        for key in self.__params:
            parameters[key] = getattr(self, key)
        #print dic
        return parameters

    #parameters = self.params

    def describe(self):
        print 'Description of ' + self.name
        print 'Parameters:', self.__params

