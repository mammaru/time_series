import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt

def myfactrial(n):
	return 1 if n==1 else n * myfactrial(n-1)

class ssm:
	def __init__(self, p, k):
		self.sys_dim = p
		self.obs_dim = k
		self.x0mean = np.matrix(np.random.randn(p,1))
		self.x0var = np.matrix(np.eye(p)) # fixed
		self.F = np.matrix(np.random.randn(k,k)) # system transition matrix
		self.Q = np.matrix(np.eye(k)) # system noise variance
		self.H = np.matrix(np.eye(p,k)) # observation transition matrix
		self.R = np.matrix(np.diag(np.random.normal(size=p))) # observation noise variance
	
	def sys_eq(self, x):
		return np.asarray(self.F * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.sys_dim]).tolist()[0], np.asarray(self.Q))).T)
	
	def obs_eq(self, x):
		return np.asarray(self.H * x + np.matrix(np.random.multivariate_normal(np.zeros([1,k]).tolist()[0], np.asarray(self.R))).T)
	
	def generate_data(self, N):
		sys_value = np.random.randn(self.sys_dim,1)
		obs_value = np.asarray(np.zeros((self.obs_dim,1)))
		i = 0
		while(i < N):
			sysi = np.matrix(sys_value)[:,i]
			sys_value = np.hstack((sys_value, sys_eq(sysi,self.F,self.Q)))
			obs_value = np.hstack((obs_value, obs_eq(sysi,self.H,self.R)))
			i += 1

		sys_value = DataFrame(sys_value.T)
		obs_value = DataFrame(obs_value.T)
		return sys_value, obs_value #return as taple object

class kalman(ssm):
	def __init__(self, p, k):
		# constant for kalman
		self.ssm = ssm(p, k)

		# variable for kalman
		self.xp = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.vp = []
		self.xf = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.vf = []
		self.xs0 = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.xs = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.vs0 = []
		self.vs = []
		self.vLag = []


	def pfs(self, obs):
		""" body of the kalman's method called prediction, filtering and smoothing """
		
		N = obs.shape[0] # number of time points
		Yobs = np.matrix(obs.T)
		
		p = self.ssm.sys_dim
		k = self.ssm.obs_dim
		x0mean = self.ssm.x0mean
		x0var = self.ssm.x0var
		
		F = self.ssm.F
		H = self.ssm.H
		Q = self.ssm.Q
		R = self.ssm.R
		xp = np.asmatrix(self.xp).T
		vp = self.vp
		xf = np.asmatrix(self.xf).T
		vf = self.vf
		xs = np.asmatrix(self.xs).T
		vs = self.vs
		vLag = self.vLag

		x0 = np.matrix(np.random.multivariate_normal(x0mean.T.tolist()[0], np.asarray(x0var))).T
		xp = np.hstack([xp, np.matrix(sys_eq(x0,F,Q))])
		vp.append(F*x0var*F.T+Q)

		for i in range(N):
			# filtering
			K = vp[i]*H.T*(H*vp[i]*H.T+R).I
			xf = np.hstack([xf, xp[:,i]+K*(Yobs[:,i]-H*xp[:,i])])
			vf.append(vp[i]-K*H*vp[i])
			# prediction  
			xp = np.hstack([xp, F*xf[:,i]])
			vp.append(F*vf[i]*F.T+Q)

		# smoothing
		J = [np.matrix(np.zeros([k,k]))]
		xs = xf[:,N-1]
		vs.insert(0, vf[N-1])
		vLag.insert(0, F*vf[N-2]-K*H*vf[N-2])
		
		for i in reversed(range(N)[1:]):
			J.insert(0, vf[i-1]*F.T*vp[i].I)
			xs = np.hstack([xf[:,i-1]+J[0]*(xs[:,0]-xp[:,i]),xs])
			vs.insert(0, vf[i-2]+J[0]*(vs[0]-vp[i])*J[0].T)
		
		for i in reversed(range(N)[2:]):
			vLag.insert(0, vf[i-1]*J[i-1].T+J[i-1]*(vLag[0]-F*vf[i-1])*J[i-2].T)
		
		J0 = x0var*F.T*vp[0].I
		vLag[0] = vf[0]*J0.T+J[0]*(vLag[0]-F*vf[0])*J0.T
		xs0 = x0mean+J0*(xs[:,0]-xp[:,0])
		vs0 = x0var+J0*(vs[0]-vp[0])*J0.T

		x0mean = xs0

		self.xp = DataFrame(xp.T)
		self.vp = vp
		self.xf = DataFrame(xf.T)
		self.vf = vf
		self.xs0 = DataFrame(xs0.T)
		self.xs = DataFrame(xs.T)
		self.vs0 = vs0
		self.vs = vs
		self.vLag = vLag
		
		return DataFrame(xs.T)

	def likelihood():
		return 1
	
	def em():
		return 1


if __name__ == "__main__":
	tmp = kalman(10,10)
	data = tmp.ssm.generate_data(20)
	result = tmp.pfs(data[1])
	ros = data[1]-result
	plt.plot(ros)
	plt.show()
