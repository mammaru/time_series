import numpy as np
from numpy.random import * 
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import VectorAutoRegressiveModel as var

SVAR0 = 1e-3
SVARthr = 1e-5

class SparseVAR(var):
	def __init__(self, p):
		self.var = var(p)
		
	def SVAR(self, lmd, itrt, flag):
		Z = t(Y[:,-1])
		X = t(Y[:,-T])
		Bold = np.inv(X.T*X+np.diag(lmd,p))*X.T*Z
		for i in range(itrt):
			Bnew = numeric(0)
			for j in range(p):
				D = np.diag(lmd/abs(Bold[:,j]))
				#bnew = invM( t(X)%*%X+(lmd*lmd)*D ) %*% t(X) %*% Z[,j]
				#bnew = invM( t(X)%*%X+(T-1)*D ) %*% t(X) %*% Z[,j]
				bnew = np.inv(X.T*X+D)*X.T*Z[:,j]
				bnew[abs(bnew)<SVAR0] = SVAR0 # sign(x)*SVARthr
				Bnew = cbind(Bnew,bnew)
			Bold = Bnew
		B = Bold
		if flag:
			B[B<SVARthr] = 0
		return B
	
	def SVAR2(self, Y, lmd, itrt, flag):
		Z = t(Y[:,-1])
		X = t(Y[:,-T])
		Bold = np.inv(X.T*X+np.diag(lmd,p))*X.T*Z
		Bnew = numeric(0)
		if lmd:
			for j in range(p):
				end_flag = 0
				z = Z[:,j]
				bold = np.matrix(Bold[:,j],length(Bold[:,j]),1,dimnames=rownames(Bold[:,j]))
				count = 0
				while not end_flag:
					count = count + 1
					#b = apply(bold,c(1,2),if0thenthr=function(x){if(abs(x)==0){x = SVARthr};return(x)})
					b[b==0] = SVARthr
					#print(b)
					D = np.diag(lmd/abs(c(b)))
					#bnew = invM( t(X)%*%X+(T-1)*D ) %*% t(X) %*% Z[,j]
					bnew = np.inv(X.T*X+D)*X.T*z
					#bnew = apply(bnew,c(1,2),ifthrthen0=function(x){if(abs(x)<=SVARthr){x = 0};return(x)})
					bnew[bnew<=SVARthr] = 0
					if sum(abs(bnew-bold))<1e-9:
						end_flag = 1
					bold = bnew
					#if(count%%10==0){cat(".")}

			#print(count)
			Bnew = np.hstack((Bnew,bnew))
			cat(".")
			B = Bnew
			
		else:
			B = Bold

		return B

	def GCV(self, lmd, itrt):
		Z = t(Y[:,-1])
		X = t(Y[:,-T])  
		B = SVAR2(Y,lmd,itrt,0)
		gcv = []
		for j in range(p):
			rss = (Z[:,j]-X*B[:,j]).T*(Z[:,j]-X*B[:,j])
			#b = apply(matrix(B[,j],length(B[,j]),1),c(1,2),if0thenthr=function(x){if(abs(x)==0){x = SVARthr};return(x)})
			b = B[:,j]
			b[b==0] = SVARthr
			D = np.diag(lmd/abs(b))
			#H = X%*%invM( t(X)%*%X+(lmd*lmd)*D )%*%t(X)
			#H = X%*%invM( t(X)%*%X+(T-1)*D )%*%t(X)
			H = X*np.inv(X.T*X+D)*X.T
			df = sum(np.diag(H))
			gcvtmp = rss/((1-(df)/(T-1))^2)
			gcvtmp = gcvtmp/(T-1)
			gcv.append(gcvtmp)
		
		sumgcv = mean(gcv)
        #sumgcv = median(gcv)  
        #cat(".")
		return sumgcv



if __name__ == "__main__":
	tmp = var(5)
	data = tmp.gen_data(10)
