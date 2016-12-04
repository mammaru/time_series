#coding: utf-8
import numpy as np
#import pandas as pd
from pandas import DataFrame
from .base import acv as fun_acv
from .base import acf as fun_acf
from .base import ccv as fun_ccv
from .base import ccf as fun_ccf



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

    # TODO: map or apply is better
    def acv(self, k):
        return fun_acv(self, k)
        
    def acf(self, k):
        return fun_acf(self, k)
    
    # TODO: map or apply is better
    def ccv(self, k):
        return fun_ccv(self, k)

    # TODO: map or apply is better
    def ccf(self, k):
        return fun_ccf(self, k)
