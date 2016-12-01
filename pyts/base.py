#coding: utf-8
import numpy as np
#import pandas as pd
from pandas import DataFrame


class TimeSeries(DataFrame):
    """ Wrapper class of DataFrame of pandas"""
    def __init__(self, x, n=None, t=None, p=None, name='TimeSeries'):
        index = n or t
        if index is not None:
            if p is not None:
                super(TimeSeries, self).__init__(x, index=index, columns=p)
            else:
                super(TimeSeries, self).__init__(x, index=index)
        else:
            if p is not None:
                super(TimeSeries, self).__init__(x, columns=p)
            else:
                super(TimeSeries, self).__init__(x)
        self.name = name

    def __str__(self):
        print 'TimeSeries:'
        return super(TimeSeries, self).__str__()

    def acv(self, k):
        mu = self.mean() # Series
        y = self.values
        N, M = self.shape
        result = []
        for m in range(M):
            tmp = 0.0
            for n in range(k+1, N):
                tmp += (y[n,m] - mu[m])*(y[n-k,m] - mu[m])
            result.append(tmp/N)
        return np.array(result)
        
    def acf(self, k):
        auto_cov = self.acv(k)
        auto_cov_0 = self.acv(0)
        return auto_cov/auto_cov_0
        

    def ccv(self, k):
        mu = self.mean() # Series
        y = self.values
        N, M = self.shape
        auto_cov = self.acv(k)
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

    def ccf(self, k):
        N, M = self.shape
        cross_cov = self.ccv(k)
        cross_cov_0 = self.ccv(0)
        result = np.diag(self.acf(k))
        for i in range(M):
            for j in range(M):
                if i!=j:
                    result[i,j] = cross_cov[i,j] / np.sqrt(cross_cov_0[i,i] * cross_cov_0[j,j])
        return result

class Parameters:
    def __init__(self):
        pass

class TimeSeriesModel(object):
    def __init__(self, **args):
        self.name = 'Time Series Model'
        self.__set_parameters(args.items())
        #self.parameters = self.params

    def __str__(self):
        return self.name

    def __set_parameters(self, parameters):
        self.__params = []
        for key, value in parameters:
            self.__params.append(key)
            setattr(self, key, value)
            #instance_attr = key
            #instance_attr[key] = value

    @property
    def params(self):
        pass

    @params.getter
    def params(self):
        parameters = {}
        for key in self.__params:
            parameters[key] = getattr(self, key)
        #print dic
        return parameters

    def describe(self):
        print 'Description of ' + self.name
        print self.params

