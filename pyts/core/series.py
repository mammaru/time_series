#coding: utf-8
from warnings import warn
from enum import Enum
import numpy as np
#import pandas as pd
from pandas import DataFrame, NaT, to_datetime
from pandas.core.common import PandasError
from pandas.tseries.index import DatetimeIndex
from pandas.util.decorators import Appender
import pyts.core.format as fmt
from pyts.core.base import acv as fun_acv
from pyts.core.base import acf as fun_acf
from pyts.core.base import ccv as fun_ccv
from pyts.core.base import ccf as fun_ccf

class Direction(Enum):
    VERTICAL = 1
    HORIZONTAL = 2
    UNDIFINED = 3


class TimeSeries(DataFrame):
    """ Wrapper class of DataFrame """
    def __init__(self, x=None, timepoints=None, name='TimeSeries'):
        try:
            super(TimeSeries, self).__init__(x)
            if isinstance(timepoints, np.ndarray):
                if timepoints.shape[0]==self.shape[0]:
                    self.index = timepoints
                elif timepoints.shape[1]==self.shape[1]:
                    self.columns = timepoints
            elif not timepoints is None:
                if len(timepoints)==self.shape[0]:
                    self.index = timepoints
                elif len(timepoints)==self.shape[1]:
                    self.columns = timepoints
                else:
                    raise                
        except PandasError as e:
            raise e
        except SyntaxError as e:
            raise e
        except:
            print('The length of timepoints is ilegal.')
        else:
            self.name = name
            self.__check_direction()

    def __setattr__(self, key, value):
        super(TimeSeries, self).__setattr__(key, value)
        if key=='index' or key=='columns':
            self.__check_direction()

    def __getattr__(self, key):
        if key=='timepoints' or key=='features':
            try:
                super(TimeSeries, self).__getattr__(key)
            except AttributeError as e:
                print "Timepoints axis is ambiguous."
                raise e
        else:
            super(TimeSeries, self).__getattr__(key)

    def __check_direction(self):
        if isinstance(self.index, DatetimeIndex) and not isinstance(self.columns, DatetimeIndex):
            self.__set_vertical()
        elif isinstance(self.columns, DatetimeIndex) and not isinstance(self.index, DatetimeIndex):
            self.__set_horizontal()
        else:
            warn('Ambiguous Definition: index or columns should be DatetimeIndex.')
            if self.__dict__.has_key('timepoints'):
                self.__dict__.pop('timepoints')
            if self.__dict__.has_key('features'):
                self.__dict__.pop('features')
            self.__time_direction = Direction.UNDIFINED

    def __set_vertical(self):
        self.__dict__['timepoints'] = self.index
        self.__dict__['features'] = self.columns
        self.__time_direction = Direction.VERTICAL

    def __set_horizontal(self):
        self.__dict__['timepoints'] = self.columns
        self.__dict__['features'] = self.index
        self.__time_direction = Direction.HORIZONTAL 


    @property
    def direction(self):
        return self.__time_direction

    #@property
    #def timepoints(self):
        #return self.__timepoints
    
    def to_dataframe(self):
        return DataFrame(self.values, index=self.index, columns=self.columns)

    def to_matrix(self):
        return np.matrix(self)

    def __str__(self):
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
        #return TimeSeries(super(TimeSeries, self).T)
        return TimeSeries(super(DataFrame, self).transpose(1, 0))

    T = property(transpose)

    def acv(self, k):
        if self.__direction == 'V':
            return fun_acv(self, k)
        elif self.__direction == 'H':
            return fun_acv(self.T, k)
        else:
            raise
        
    def acf(self, k):
        return fun_acf(self, k)
    
    def ccv(self, k):
        return fun_ccv(self, k)

    def ccf(self, k):
        return fun_ccf(self, k)
