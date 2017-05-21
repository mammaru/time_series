#coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyts.ssm import DynamicLinearModel as DLM
from pyts.example import exchange
from pyts.em import EM
from pyts.util import normalize, heatmap
from pyts.core.series import TimeSeries as ts

if __name__ == "__main__":
    if 1:
        ex = exchange()
        price = ex["price"]
        volume = ex["volume"]

    # SVAR
    if 0:
        data = exprs
        svar = SparseVAR()
        svar.set_data(data)

        interval = np.arange(0,3,0.5)
        B = svar.GCV(interval)
        B = DataFrame(B.T, index=data.columns, columns=data.columns)

    if 1:
        tmp = ts(price)
        #print tmp
        tmp.describe()

    # SSM
    if 1:
        data = normalize(price[[0,1,2,4,5]])
        #print np.mean(data), np.std(data)
        dlm =  DLM(observation_dimention=data.shape[1],system_dimention=3)
        dlm.describe()
        #print params
        #for key, value in dlm.params:
            #print key, value
        em = EM()
        em(dlm, data)
        #em.execute(dlm, data)

    if 1:
        heatmap(dlm.H)
    if 0:
        digraph(dlm.F)
