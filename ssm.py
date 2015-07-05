import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt

def myfactrial(n):
	return 1 if n==1 else n * myfactrial(n-1)

# create data
data = DataFrame(np.random.randn(10,4),columns=["a","b","c","d"])
sys0 = DataFrame(np.random.randn(1,4),columns=["a","b","c","d"])

matF = np.matrix(np.random.randn(4,4))
matQ = np.matrix(np.random.randn(4,4))
matH = np.matrix(np.random.randn(4,4))
matR = np.matrix(np.random.randn(4,4))

T = 20

def sys_eq(x,F,Q):
	dim = x.shape[0]
	return np.asarray(F * x + np.random.normal(size=(dim,1)))
  
def obs_eq(x,H,R):
	dim = x.shape[0]
	return np.asarray(H * x + np.random.normal(size=(dim,1)))

sys_value = np.asarray(sys0.T)
obs_value = np.asarray(np.zeros((4,1)))
i = 0
while(i < T):
	sysi = np.matrix(sys_value)[:,i]
	sys_value = np.hstack((sys_value, sys_eq(sysi,matF,matQ)))
	obs_value = np.hstack((obs_value, obs_eq(sysi,matH,matR)))
	i += 1

sys_value = DataFrame(sys_value.T)
obs_value = DataFrame(obs_value.T)



class SSM:
	def __init__(self, p, k):
		self.sys_dim = p
		self.obs_dim = k
		self.x0mean = np.matrix(np.random.randn(p,1))
		self.x0var = np.matrix(np.identity(p)) # fixed
		self.F = np.matrix(np.random.randn(k,k))
		self.H = np.matrix(np.eye(p,k))
		self.Q = np.matrix(np.eye(k))
		self.R = np.matrix(np.diag(np.random.normal(size=p)))

	#def test1(self):
		#F = self.F
		#print F
		
		self.xf = numeric(0)
		self.vf = as.list(NULL)
		self.xs0 = numeric(0)
		self.xs = numeric(0)
		self.vs0 = numeric(0)
		self.vs = as.list(NULL)
		self.vLag = as.list(NULL)

	#def set_params:
		
	def kf(maxT, Yobs):

		X0mean = self.x0mean
		X0var = self.x0var
		F = self.f
		H = self.h
		Q = self.q
		R = self.r

		x0 = np.matrix(np.random.multivariate_normal(mp.asarray(X0mean.T), np.asarray(X0var)))
		xPri = np.matrix(sys_eq(x0,F,Q))
		sigmaPri = [F * X0var * F.T + Q]

		for(i in np.arange(maxT)){
		  #filtering
		  K = sigmaPri[i]*H.T*(H%*%result$sigmaPri[[i]]%*%t(H) + R).I
		  xPost = cbind(result$xPost,result$xPri[,i] + K%*%(Yobs[,i] - H%*%result$xPri[,i]))
		  result$sigmaPost[[i]] = result$sigmaPri[[i]] - K%*%H%*%result$sigmaPri[[i]]

		  #prediction  
		  result$xPri = cbind(result$xPri,F%*%result$xPost[,i])
		  result$sigmaPri[[i+1]] = F%*%result$sigmaPost[[i]]%*%t(F) + Q

		}

		#smoothing
		J = as.list(NULL)
		J[[maxT]] = matrix(rep(0,k*k),k,k)
		result$xT = matrix(result$xPost[,maxT],k,1)
		result$sigmaT = as.list(NULL)
		result$sigmaT[[maxT]] = result$sigmaPost[[maxT]]
		result$sigmaLag = as.list(NULL)
		result$sigmaLag[[maxT]] = F%*%result$sigmaPost[[maxT-1]] - K%*%H%*%result$sigmaPost[[maxT-1]]

		for(i in maxT:2){
		  J[[i-1]] = result$sigmaPost[[i-1]]%*%t(F)%*%invM(result$sigmaPri[[i]])
		  result$xT = cbind(result$xPost[,i-1] + J[[i-1]]%*%(result$xT[,1]-result$xPri[,i]),result$xT)
		  result$sigmaT[[i-1]] = result$sigmaPost[[i-1]] + J[[i-1]]%*%(result$sigmaT[[i]]-result$sigmaPri[[i]])%*%t(J[[i-1]])
		}
		for(i in maxT:3){
		  result$sigmaLag[[i-1]] = result$sigmaPost[[i-1]]%*%t(J[[i-1]]) + J[[i-1]]%*%(result$sigmaLag[[i]]-F%*%result$sigmaPost[[i-1]])%*%t(J[[i-2]])
		}
		J0 = X0var%*%t(F)%*%invM(result$sigmaPri[[1]])
		result$sigmaLag[[1]] = result$sigmaPost[[1]]%*%t(J0) + J[[1]]%*%(result$sigmaLag[[2]]-F%*%result$sigmaPost[[1]])%*%t(J0)
		result$xT0 = X0mean + J0%*%(result$xT[,1]-result$xPri[,1])
		result$sigmaT0 = X0var + J0%*%(result$sigmaT[[1]]-result$sigmaPri[[1]])%*%t(J0)

		result$x0mean = result$xT0

		return result

	def likelihood():
		return 1
	
	def em():
		return 1

