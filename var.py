import numpy as np
from numpy.random import * 
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import VectorAutoRegressiveModel as var

SVAR0 = 1e-7
SVARthr = 1e-7

class SparseVAR(var):
	def __init__(self):
		self.dim = 0 # dimention of vector
		#self.var = var(p) # make instance of vector auto regressive model
		self.lmd = 0

	def set_data(self, data):
		self.data = data
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
		X = np.matrix(self.data.ix[:(N-2),])
		Bold = (X.T*X+np.matrix(np.diag([lmd for i in range(p)]))).I*X.T*Z
		Bnew = np.matrix(np.eye(p, 0))
		if lmd:
			for j in range(p):
				end_flag = 0
				z = Z[:,j]
				bold = Bold[:,j]
				count = 0
				while not end_flag:
					#print ".",
					count = count + 1
					b = bold
					b[b==0] = SVARthr
					D = np.matrix(np.diag((lmd/np.abs(b.T)).tolist()[0]))
					#bnew = invM( t(X)%*%X+(T-1)*D ) %*% t(X) %*% Z[,j]
					bnew = (X.T*X+D).I*X.T*z
					#bnew = apply(bnew,c(1,2),ifthrthen0=function(x){if(abs(x)<=SVARthr){x = 0};return(x)})
					bnew[bnew<=SVARthr] = 0
					#print np.sum(np.abs(bnew-bold))
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
		#return self.Ahat

	def GCV(self):
		p = self.dim
		N = self.N
		Z = np.matrix(self.data.ix[1:,])
		X = np.matrix(self.data.ix[:(N-2),])

		gcvloss = []
		for lmd in range(20):
			print "lambda: ", lmd, "gcvloss:",
			self.SVAR(lmd)
			B = self.Ahat.T
			gcv = []
			for j in range(p):
				#print ".",
				rss = (Z[:,j]-X*B[:,j]).T*(Z[:,j]-X*B[:,j])
				#b = apply(matrix(B[,j],length(B[,j]),1),c(1,2),if0thenthr=function(x){if(abs(x)==0){x = SVARthr};return(x)})
				b = B[:,j]
				b[b==0] = SVARthr
				#D = np.diag(lmd/np.abs(b))
				D = np.matrix(np.diag((lmd/np.abs(b.T)).tolist()[0]))
				#H = X%*%invM( t(X)%*%X+(lmd*lmd)*D )%*%t(X)
				#H = X%*%invM( t(X)%*%X+(T-1)*D )%*%t(X)
				H = X*(X.T*X+D).I*X.T
				df = np.sum(np.diag(H))
				gcvtmp = rss/((1-(df)/(N-1))**2)
				gcvtmp /= N-1
				gcv.append(gcvtmp)
				sumgcv = np.mean(gcv)
				#sumgcv = median(gcv)  
				#cat(".")
			print sumgcv
			gcvloss.append(sumgcv)

		self.gcvloss = gcvloss
		#return 1



if __name__ == "__main__":
	tmp = var(100)
	data = tmp.gen_data(20)

	svar = SparseVAR()
	svar.set_data(data)
	#svar.SVAR(5)
	svar.GCV()
