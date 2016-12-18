#coding: utf-8
import numpy as np
#import pandas as pd
from pandas import DataFrame
from pandas.tseries.index import DatetimeIndex
from .base import acv as fun_acv
from .base import acf as fun_acf
from .base import ccv as fun_ccv
from .base import ccf as fun_ccf



class TimeSeries(DataFrame):
    """ Wrapper class of DataFrame """
    def __init__(self, x=None, n=None, t=None, p=None, name='TimeSeries'):
        index = n or t
        super(TimeSeries, self).__init__(x, index=index, columns=p)
        self.name = name

    def __check_vertical(self):
        if isinstance(self.index, DatetimeIndex) or not isinstance(self.columns, DatetimeIndex):
            return True
        else:
            return False

    def __check_horizontal(self):
        if isinstance(self.columns, DatetimeIndex):
            return True
        else:
            return False

    def __ensure_direction(self):
        if __check_vertical():
            self.timepoints = self.index
        elif __check_horizontal():
            self.timepoints = self.columns

    def to_dataframe(self):
        return DataFrame(self.values, index=self.index, columns=self.columns)

    def to_matrix(self):
        return np.matrix(self)

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
