#coding: utf-8
import numpy as np
#import pandas as pd
from pandas import DataFrame
from pandas.tseries.index import DatetimeIndex
from pandas.util.decorators import Appender
import pyts.core.format as fmt
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
        self.__check_direction()

    def __check_vertical(self):
        if isinstance(self.index, DatetimeIndex): #or not isinstance(self.columns, DatetimeIndex):
            return True
        else:
            return False

    def __check_horizontal(self):
        if isinstance(self.columns, DatetimeIndex):
            return True
        else:
            return False

    def __check_direction(self):
        if isinstance(self.index, DatetimeIndex): #or not isinstance(self.columns, DatetimeIndex):
            self.__direction = 'V'
        elif isinstance(self.columns, DatetimeIndex):
            self.__direction = 'H'
        else:
            self.__direction = undifined

    @property
    def timepoints(self):
        if self.__check_vertical():
            return self.index 
        elif self.__check_horizontal():
            return self.columns
        else:
            return undifined

    def to_dataframe(self):
        return DataFrame(self.values, index=self.index, columns=self.columns)

    def to_matrix(self):
        return np.matrix(self)

    def __str__(self):
        print self.name + ':'
        return super(TimeSeries, self).__str__()

    @Appender(fmt.docstring_to_string, indents=1)
    def to_string(self, buf=None, columns=None, col_space=None, colSpace=None,
                  header=True, index=True, na_rep='NaN', formatters=None,
                  float_format=None, sparsify=None, index_names=True,
                  justify=None, line_width=None, max_rows=None, max_cols=None,
                  show_dimensions=False):
        """
        Render a TimeSeries to a console-friendly tabular output.
        """

        if colSpace is not None:  # pragma: no cover
            warnings.warn("colSpace is deprecated, use col_space",
                          FutureWarning)
            col_space = colSpace

        formatter = fmt.TimeSeriesFormatter(self, buf=buf, columns=columns,
                                            col_space=col_space, na_rep=na_rep,
                                            formatters=formatters,
                                            float_format=float_format,
                                            sparsify=sparsify,
                                            justify=justify,
                                            index_names=index_names,
                                            header=header, index=index,
                                            line_width=line_width,
                                            max_rows=max_rows,
                                            max_cols=max_cols,
                                            show_dimensions=show_dimensions)
        formatter.to_string()

        if buf is None:
            result = formatter.buf.getvalue()
            return result


    def transpose(self):
        """Transpose index and columns"""
        return TimeSeries(super(TimeSeries, self).T)
        #return super(DataFrame, self).transpose(1, 0)

    T = property(transpose)

    def acv(self, k):
        return fun_acv(self, k)
        
    def acf(self, k):
        return fun_acf(self, k)
    
    def ccv(self, k):
        return fun_ccv(self, k)

    def ccf(self, k):
        return fun_ccf(self, k)
