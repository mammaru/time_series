import numpy as np
from numpy.random import *
#from numpy import nan as NaN
#import pandas as pd
from pandas import DataFrame, Series
#from matplotlib import pyplot as plt

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




class StateSpaceModel:
	def __init__(self, p, k):
		self.obs_dim = p
		self.sys_dim = k
		self.x0mean = np.matrix(np.random.randn(k, 1))
		self.x0var = np.matrix(np.eye(k)) # fixed
		self.F = np.matrix(np.random.randn(k,k)) # system transition matrix
		#self.F = np.matrix(DataFrame(self.F).applymap(lambda x: 0 if np.abs(x)>0.5 else x))
		self.F[abs(self.F)<1] = 0
		self.Q = np.matrix(np.eye(k)) # system noise variance
		self.H = np.matrix(np.eye(p,k)) # observation transition matrix
		self.R = np.matrix(np.diag(np.diag(np.random.rand(p,p)))) # observation noise variance
	
	def sys_eq(self, x, F, Q):
		return np.asarray(F * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.sys_dim]).tolist()[0], np.asarray(Q))).T)
	
	def obs_eq(self, x, H, R):
		return np.asarray(self.H * x + np.matrix(np.random.multivariate_normal(np.zeros([1,self.obs_dim]).tolist()[0], np.asarray(self.R))).T)
	
	def gen_data(self, N):
		sys_value = np.random.randn(self.sys_dim,1)
		obs_value = np.asarray(np.zeros((self.obs_dim,1)))
		i = 0
		while(i < N):
			sysi = np.matrix(sys_value)[:,i]
			sys_value = np.hstack((sys_value, self.sys_eq(sysi, self.F, self.Q)))
			obs_value = np.hstack((obs_value, self.obs_eq(sysi, self.H, self.R)))
			i += 1
		sys_value = DataFrame(sys_value.T)
		obs_value = DataFrame(obs_value.T)
		return sys_value, obs_value #return as taple object


class Kalman(StateSpaceModel):
	def __init__(self, p, k):
		# constant for kalman
		self.ssm = StateSpaceModel(p, k)
		# variable for kalman
		self.x0mean = self.ssm.x0mean
		self.x0var = self.ssm.x0var
		self.F = self.ssm.F
		self.H = self.ssm.H
		self.Q = self.ssm.Q
		self.R = self.ssm.R
		self.xp = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.vp = []
		self.xf = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.vf = []
		self.xs0 = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.xs = DataFrame(np.empty([self.ssm.sys_dim, 0])).T
		self.vs0 = []
		self.vs = []
		self.vLag = []

	def set_data(self, data):
		self.obs = data
		self.unequal_intarval_flag = True if sum(np.sum(data)) else False
		self.missing_data_flag = True if sum(np.sum(data)) else False

	def set_pfs_results(self):
		Yobs = np.matrix(self.obs.T)
		N = self.obs.shape[0] # number of time points
		
		xs0 = np.matrix(self.xs0.T)
		xp = np.matrix(self.xp.T)
		vp = self.vp
		xf = np.matrix(self.xf.T)
		vf = self.vf
		xs0 = np.matrix(self.xs0.T)
		xs = np.matrix(self.xs.T)
		vs0 = self.vs0
		vs = self.vs
		vLag = self.vLag

		S11 = xs[:,0]*xs[:,0].T + vs[0]
		S10 = xs[:,0]*xs0.T + vLag[0]
		S00 = xs0*xs0.T + x0var
		Syy = Yobs[:,0]*Yobs[:,0].T
		Syx = Yobs[:,0]*xs[:,0].T
		for i in range(N)[1:]:
			S11 += xs[:,i-1]*xs[:,i-1].T + vs[i-1]
			S10 += xs[:,i-1]*xs[:,i-2].T + vLag[i-1]
			S00 += xs[:,i-2]*xs[:,i-2].T + vs[i-2]
			Syy += Yobs[:,i-1]*Yobs[:,i-1].T
			Syx += Yobs[:,i-1]*xs[:,i-1].T
		
		self.S11 = S11
		self.S10 = S10
		self.S00 = S00
		self.Syy = Syy
		self.Syx = Syx
		

	def pfs(self):
		""" body of the kalman's method called prediction, filtering and smoothing """
		
		N = self.obs.shape[0] # number of time points
		Yobs = np.matrix(self.obs.T)
		
		p = self.ssm.obs_dim
		k = self.ssm.sys_dim
		x0mean = self.x0mean
		x0var = self.x0var
		
		F = self.F
		H = self.H
		Q = self.Q
		R = self.R
		#xp = np.matrix(np.empty([k, 0])).T #np.matrix(self.xp.T)
		vp = self.vp
		#xf = np.matrix(self.xf.T)
		vf = self.vf
		#xs = np.matrix(np.empty([k, 0])).T #np.matrix(self.xs.T)
		vs = self.vs
		vLag = self.vLag

		#x0 = np.matrix(np.random.multivariate_normal(x0mean.T.tolist()[0], np.asarray(x0var))).T
		#xp = np.matrix(self.ssm.sys_eq(x0,F,Q))
		xp = np.matrix(F*x0mean)
		vp.append(F*x0var*F.T+Q)

		if 0: # unequal intervals
			interval = tp[2]-tp[1]
			maxt = tp[maxT]/interval
			t = tp[2]
			j = 1
			xP = F*x0mean
			vP = list(F*self.vPost0*F.T + Q)
			#x = self.xPost
			#v = self.vPost
			for i in range(maxt+1):
				if interval*(i-1)==tp[j]: # obs exists
					self.xPri = cbind(self.xPri,xP[:,i])
					self.vPri[j] = vP[i]

					#filtering
					K = vP[i]*H.T*(H*vP[i]*H.T + R).I
					x = xP[:,i] + K*(Yobs[:,j] - H*xP[:,i])
					v = vP[i] - K*H*vP[i]
					self.xPost = cbind(self.xPost,x)
					self.vPost[j] = v
					j = j+1
				else: # obs does not exist
					x = xP[:,i]
					v = vP[i]
					
				#prediction
				xP = cbind(xP,F*x)
				vP[i+1] = F*v*F.T + Q
			
			self.x = xP
			self.v = vP

		else: # equal intervals
			for i in range(N):
				# filtering
				K = vp[i]*H.T*(H*vp[i]*H.T+R).I
				xf = xp[:,i]+K*(Yobs[:,i]-H*xp[:,i]) if i == 0 else np.hstack([xf, xp[:,i]+K*(Yobs[:,i]-H*xp[:,i])])
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
		
		self.xs0 = DataFrame(xs0.T)
		self.xp = DataFrame(xp.T)
		self.vp = vp
		self.xf = DataFrame(xf.T)
		self.vf = vf
		self.xs0 = DataFrame(xs0.T)
		self.xs = DataFrame(xs.T)
		self.vs0 = vs0
		self.vs = vs
		self.vLag = vLag

