#coding: utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from var.var import *
from ssm.kalman import *
from util import *

if __name__ == "__main__":
    if 1:
        filename = "./data/exchange.dat"
        df = pd.read_table(filename, index_col="datetime")
        df.index = pd.to_datetime(df.index) # convert index into datetime
        #hourly = df.resample("H", how="mean") # hourly
        daily = df.resample("D", how="mean") # daily
        price = daily.ix[:, daily.columns.map(lambda x: x.endswith("PRICE"))]
        volume = daily.ix[:, daily.columns.map(lambda x: x.endswith("VOLUME"))]

    # SVAR
    if 1:
        data = exprs
        svar = SparseVAR()
        svar.set_data(data)

        interval = np.arange(0,3,0.5)
        B = svar.GCV(interval)
        B = DataFrame(B.T, index=data.columns, columns=data.columns)
    if 1:
        heatmap(B)
    if 0:
        digraph(B)

    # kalman and EM
    if 0:
        sys_k = 5
        em = EM(price, sys_k)
        em.execute()

    if 0:
        fig, axes = plt.subplots(np.int(np.ceil(sys_k/3.0)), 3, sharex=True)
        j = 0
        for i in range(3):
            while j<sys_k:
                if sys_k<=3:
                    axes[j%3].plot(data[0][j], "k--", label="obs")
                    axes[j%3].plot(em.kl.xp[j], label="prd")
                    axes[j%3].legend(loc="best")
                    axes[j%3].set_title(j)
                else:
                    axes[i, j%3].plot(data[0][j], "k--", label="obs")
                    axes[i, j%3].plot(em.kl.xp[j], label="prd")
                    axes[i, j%3].legend(loc="best")
                    axes[i, j%3].set_title(j)
                j += 1
                if j%3 == 2: break

        fig.show()
		
        #loss = data[0]-em.kl.xs
        #plt.plot(loss)
        #plt.plot(em.llh)
        #plt.show()
