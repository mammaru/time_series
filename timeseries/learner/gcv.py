#coding: utf-8
import sys
import numpy as np
from numpy.random import * 
from pandas import DataFrame, Series
from matplotlib import pyplot as plt

SVARthr = 1e-5
SVAR0 = 1e-20
MAX_INT = sys.maxint

def GCV(self, interval):
    p = self.dim
    N = self.N
    Z = np.matrix(self.data.ix[1:,])
    X = np.matrix(self.data.ix[:(N-1),])

    max_sumgcv = MAX_INT
    gcvloss = Series(index=interval)
    for lmd in interval:
        print "lambda: ", lmd, "gcvloss:",
        self.regression(lmd)
        B = self.Ahat.T
        gcv = []
        for j in range(p):
            rss = (Z[:,j]-X*B[:,j]).T*(Z[:,j]-X*B[:,j])
            b = B[:,j]
            b[b==0] = SVAR0
            D = np.matrix(np.diag((lmd/np.abs(b.T)).tolist()[0]))
            H = X*(X.T*X+D).I*X.T
            d_free = np.sum(np.diag(H))
            gcvtmp = rss/((1-(d_free)/(N-1))**2)*(N-1)
            gcv.append(gcvtmp)
            sumgcv = np.mean(gcv)
            #sumgcv = median(gcv)  
            #cat(".")
        print sumgcv
        if sumgcv < max_sumgcv:
            B_best = B
            max_sumgcv = sumgcv
        gcvloss[lmd] = sumgcv

    self.gcvloss = gcvloss
    B_best[B_best<SVARthr] = 0
    return B_best


if __name__ == "__main__":
    print "svar.py: Called in main process."
    if 0:
        tmp = VectorAutoRegressiveModel(100)
        data = tmp.gen_data(20)

        cvar = ConstraintVAR()
        cvar.set_data(data)
        #cvar.regression(5)
        cvar.GCV()
