import sys
import numpy as np
from numpy.random import * 
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import VectorAutoRegressiveModel as VAR

SVAR0 = 1e-7
SVARthr = 1e-12
MAX_INT = sys.maxint

class SparseVAR(var):
	def __init__(self):
		self.dim = 0 # dimention of vector
		#self.var = VAR(p) # make instance of vector auto regressive model
		self.lmd = 0

	def set_data(self, data):
		self.data = data # input data is DataFrame
		self.N = data.shape[0] # number of time points
		self.dim = data.shape[1]

	def SVAR(self, *lmd):
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
					#b[b==0] = SVAR0
					D = np.matrix(np.diag((lmd/np.abs(b.T)).tolist()[0]))
					#bnew = invM( t(X)%*%X+(T-1)*D ) %*% t(X) %*% Z[,j]
					bnew = (X.T*X+D).I*X.T*z
					#bnew = apply(bnew,c(1,2),ifthrthen0=function(x){if(abs(x)<=SVARthr){x = 0};return(x)})
					bnew[bnew<=SVARthr] = 0
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


	def GCV(self):
		p = self.dim
		N = self.N
		Z = np.matrix(self.data.ix[1:,])
		X = np.matrix(self.data.ix[:(N-1),])

		max_sumgcv = MAX_INT
		lmd_intarval = np.arange(0,10,1)
		gcvloss = Series(index=lmd_intarval)
		for lmd in lmd_intarval:
			print "lambda: ", lmd, "gcvloss:",
			self.SVAR(lmd)
			B = self.Ahat.T
			gcv = []
			for j in range(p):
				rss = (Z[:,j]-X*B[:,j]).T*(Z[:,j]-X*B[:,j])
				b = B[:,j]
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
		return B_best



if __name__ == "__main__":
	tmp = VAR(100)
	data = tmp.gen_data(20)

	svar = SparseVAR()
	svar.set_data(data)
	#svar.SVAR(5)
	svar.GCV()
