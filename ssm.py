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
		self.x0var = np.matrix(np.eye(p)) # fixed
		self.F = np.matrix(np.random.randn(k,k)) # system transition matrix
		self.Q = np.matrix(np.eye(k)) # system noise variance
		self.H = np.matrix(np.eye(p,k)) # observation transition matrix
		self.R = np.matrix(np.diag(np.random.normal(size=p))) # observation noise variance

		self.xf = np.empty([self.sys_dim, 0])
		self.vf = []
		self.xs0 = np.empty([0,self.sys_dim, 0])
		self.xs = np.empty([0,self.sys_dim, 0])
		self.vs0 = []
		self.vs = []
		self.vLag = []

	#def set_params:
		
	def kf(self, obs):

		N = obs.shape[0]
		Yobs = np.matrix(obs.T)
		p = self.sys_dim
		k = self.obs_dim
		x0mean = self.x0mean
		x0var = self.x0var
		F = self.F
		H = self.H
		Q = self.Q
		R = self.R
		xf = self.xf
		vf = self.vf

		x0 = np.matrix(np.random.multivariate_normal(x0mean.T.tolist()[0], np.asarray(x0var))).T
		xp = np.matrix(sys_eq(x0,F,Q))
		vp = [F * x0var * F.T + Q]

		for i in range(N):
			print i
			# filtering
			K = vp[i] * H.T * (H * vp[i] * H.T + R).I
			xf = np.hstack([xf, xp[:,i] + K * (Yobs[:,i] - H * xp[:,i])])
			vf.append(vp[i] - K * H * vp[i])
			# prediction  
			xp = np.hstack([xp,F * xf[:,i]])
			vp.append(F * vf[i] * F.T + Q)

		# smoothing
		J = []
		J.insert(0, np.matrix(np.zeros([k,k])))
		xs = xf[:,N-1]
		vs = []
		vs.insert(0, vf[N-1])
		vLag = []
		vLag.insert(0, F * vf[N-2] - K * H * vf[N-2])
		
		for i in reversed(range(N)[1:]):
			print i
			J.insert(0, vf[i-1] * F.T * vp[i].I)
			print J[0]* (xs[:,0]-xp[:,i])
			xs = np.hstack([xf[:,i-1] + J[0] * (xs[:,0]-xp[:,i]),xs])
			vs.insert(0, vf[i-2] + J[0] * (vs[0]-vp[i]) * J[0].T)
		
		for i in reversed(range(N)[2:]):
			print i
			vLag.insert(0, vf[i-1] * J[i-1].T + J[i-1] * (vLag[0]-F * vf[i-1]) * J[i-2].T)
		
		J0 = x0var * F.T * vp[0].I
		vLag[0] = vf[0] * J0.T + J[0] * (vLag[0]-F * vf[0]) * J0.T
		xs0 = x0mean + J0 * (xs[:,0]-xp[:,0])
		vs0 = x0var + J0 * (vs[0]-vp[0]) * J0.T

		x0mean = xs0

		return xs

	def likelihood():
		return 1
	
	def em():
		return 1

