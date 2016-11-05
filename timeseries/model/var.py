#coding: utf-8
import sys
import numpy as np
from numpy.random import * 
from pandas import DataFrame, Series
from matplotlib import pyplot as plt

SVARthr = 1e-5
SVAR0 = 1e-20
MAX_INT = sys.maxint

class VectorAutoRegressiveModel:
    def __init__(self, p):
        self.dim = p
        self.A = np.matrix(normal(size=(p,p)))
        self.A[np.abs(self.A)>0.7] = 0 # make A sparse matrix
        self.E = np.matrix(np.diag(rand(p)))
        self.y0mean = DataFrame(np.array(rand(p))).T

    def set_param(self, *parameters):
        self.A = parameters[0]
        if parameters[1]: self.E = parameters[1]
        if parameters[2]: self.y0mean = parameters[2]

    def gen_data(self, N):
        y = np.matrix(self.y0mean.T)
        for i in range(N-1):
            e = np.matrix(np.random.multivariate_normal(np.zeros((1,self.dim)).tolist()[0], np.array(self.E))).T
            y = np.matrix(np.hstack((np.array(y), np.array(self.A*y[:,i]+e))))
        return DataFrame(y.T)

class ConstraintVAR(VectorAutoRegressiveModel):
    def __init__(self):
        self.dim = 0 # dimention of vector
        #self.var = VectorAutoRegressiveModel(p) # make instance of vector auto regressive model
        self.lmd = 0

    def set_data(self, data):
        self.data = data # input data is DataFrame
        self.N = data.shape[0] # number of time points
        self.dim = data.shape[1]

    def regression(self, *lmd):
        p = self.dim
        N = self.N
        if lmd:
            lmd = lmd[0]
        else:
            lmd = 0
        Z = np.matrix(self.data.ix[1:,])
        X = np.matrix(self.data.ix[:(N-1),])
        #print X.shape, Z.shape
        Bold = (X.T*X+np.matrix(np.diag([lmd for i in range(p)]))).I*X.T*Z
        Bnew = np.matrix(np.eye(p, 0))
        if lmd:
            for j in range(p):
                end_flag = 0
                z = Z[:,j]
                bold = Bold[:,j]
                count = 0
                while not end_flag:
                    count = count + 1
                    b = bold
                    b[b==0] = SVAR0
                    D = np.matrix(np.diag((lmd/np.abs(b.T)).tolist()[0]))
                    #print D
                    #bnew = invM( t(X)%*%X+(T-1)*D ) %*% t(X) %*% Z[,j]
                    bnew = (X.T*X+D).I*X.T*z
                    #bnew = apply(bnew,c(1,2),ifthrthen0=function(x){if(abs(x)<=SVARthr){x = 0};return(x)})
                    bnew[bnew<SVARthr] = 0
                    if np.sum(np.abs(bnew-bold))<1e-5:
                        end_flag = 1
                    elif count > 30:
                        #self.Ahat = Bnew.T
                        #break
                        end_flag = 1
                    bold = bnew
                    #if(count%%10==0){cat(".")}

                Bnew = np.hstack([Bnew,bnew])
            #print "."
            B = Bnew
            
        else:
            B = Bold

        self.Ahat = B.T


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
