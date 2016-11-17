#coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyts.ssm.kalman import StateSpaceModel as SSM
from pyts.example.data import exchange

if __name__ == "__main__":
    if 1:
        data = exchange()

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

    # SSM
    if 1:
        ssm =  SSM(2,2)
