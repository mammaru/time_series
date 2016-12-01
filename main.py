#coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyts.ssm import SSMKalman as SSM
from pyts.example.data import exchange
from pyts.em import EM
from pyts.util import normalize
from pyts.base import TimeSeries as ts

if __name__ == "__main__":
    if 1:
        data = exchange()
        price = data["price"]
        volume = data["volume"]

    # SVAR
    if 0:
        data = exprs
        svar = SparseVAR()
        svar.set_data(data)

        interval = np.arange(0,3,0.5)
        B = svar.GCV(interval)
        B = DataFrame(B.T, index=data.columns, columns=data.columns)
    if 0:
        heatmap(B)
    if 0:
        digraph(B)

    if 1:
        tmp = ts(price)
        #print tmp
        tmp.describe()

    # SSM
    if 0:
        data = normalize(price[[0,1,2,4,5]])
        #print np.mean(data), np.std(data)
        ssm =  SSM(observation_dimention=data.shape[1],system_dimention=2)
        ssm.describe()
        #print params
        #for key, value in ssm.params:
            #print key, value
        em = EM()
        em(ssm, data)
        #em.execute(ssm, data)
