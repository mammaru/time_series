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
		self.x0mean = self.ssm.x0mean
		self.x0var = self.ssm.x0var
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

	def pfs(self):
		""" body of the kalman's method called prediction, filtering and smoothing """
		
		N = self.obs.shape[0] # number of time points
		Yobs = np.matrix(self.obs.T)
		
		p = self.ssm.sys_dim
		k = self.ssm.obs_dim
		x0mean = self.x0mean
		x0var = self.x0var
		
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
		
		self.S11 = S11
		self.S10 = S10
		self.S00 = S00
		self.Syy = Syy
		self.Syx = Syx
		
		return #DataFrame(xs.T)

class Expectation_Maximization:
	def __init__(self, data, sys_dim):
		# kalman instance for em
		self.data = data
		self.p = sys_dim
		self.k = data.shape[1]
		self.N = data.shape[0]
		
		self.kl = kalman(self.p, self.k)
		# variable for em
		self.F = self.kl.ssm.F
		self.Q = self.kl.ssm.Q
		self.H = self.kl.ssm.H
		self.R = self.kl.ssm.R
		self.llh = [0]
		
	def __Estep(self):
		""" Private method: Expectation step of EM algorithm for SSM """

		# execute kalman's algorithm(prediction, filtering and smoothing)
		Yobs = np.matrix(self.data.T)
		self.kl.set_data(self.data)
		self.kl.pfs()

		p = self.p
		k = self.k
		N = self.N
		x0var = self.kl.x0var
		F = self.F
		H = self.H
		Q = self.Q
		R = self.R
		xs = np.matrix(self.kl.xs.T)
		xs0 = np.matrix(self.kl.xs0.T)
		x0mean = self.kl.x0mean
		x0var = self.kl.x0var
		vs = self.kl.vs
		vs0 = self.kl.vs0
		S11 = self.kl.S11
		S10 = self.kl.S10
		S00 = self.kl.S00
		Syy = self.kl.Syy
		Syx = self.kl.Syx

		#print R.I
		#tmp = np.matrix(np.zeros([p,p]))
		#for i in range(N):
			#tmp += (Yobs[:,i]-H*xs[:,i])*(Yobs[:,i]-H*xs[:,i]).T + H*vs[i]*H.T

		#likelihood = (-1/2)*(log(det(x0var)) + trace(x0var.I*(vs0+(xs0-x0mean)*(xs0-x0mean).T)) + N*log(det(Q)) + trace(Q.I*(S11-S10*F.T-F*S10.T+F*S00*F.T)) + N*log(det(R)) + trace(R.I*tmp) + (k+N*(k+p))*log(2*pi))

		likelihood = log(det(x0var)) + trace(inv(x0var)*(vs0+(xs0-x0mean)*(xs0-x0mean).T)) + N*log(det(R)) + trace(inv(R)*(Syy+H*S11*H.T-Syx*H.T-H*Syx.T)) + N*log(det(Q)) + trace(inv(Q)*(S11+F*S00*F.T-S10*F.T-F*S10.T)) + (k+N*(k+p))*log(2*pi)

		print N*log(det(R))
		likelihood = (-1/2)*likelihood
		print likelihood
		self.llh.append(likelihood)


	def __Mstep(self):
		""" Private method: Maximization step of EM algorithm for SSM """
		Yobs = np.matrix(self.data.T)

		p = self.p
		k = self.k
		N = self.N
		#F = self.F
		H = self.H
		Q = self.Q
		S11 = self.kl.S11
		S10 = self.kl.S10
		S00 = self.kl.S00
		Syy = self.kl.Syy
		Syx = self.kl.Syx
		xs = np.matrix(self.kl.xs.T)
		xs0 = self.kl.xs0
		vs = self.kl.vs
		
		self.F = S10*S00.I
		self.H = Syx*S11.I
		self.Q = (S11 - S10*S00.I*S10.T)/N

		tmp = np.matrix(np.zeros([p,p]))
		rtmp = 0
		for i in range(N):
			#print xs[:,i]
			tmp += (Yobs[:,i]-H*xs[:,i])*(Yobs[:,i]-H*xs[:,i]).T + H*vs[i]*H.T
			rtmp += trace((Yobs[:,i]-H*xs[:,i])*(Yobs[:,i]-H*xs[:,i]).T + H*vs[i]*H.T)
			
		rtmp = rtmp/(N*p)
		
		self.R = tmp/N

		self.x0mean = xs0
        #self.x0var = sigmaT0

        #modification for CSSM
		#if 0:
			#Q = diag(1,m)
			#self.r = (1/maxT)*diag(c(diag(self.Syy - self.Syx*invM(self.S11)*t(self.Syx))),p)
            #self.r = (1/maxT)*diag(diag(self.Syy - self.Syx*invM(self.S11)*t(self.Syx)),p)
		    #self.r = diag(rtmp,p)

		#return 1

	def execute(self):
		""" Execute EM algorithm """

		count = 0
		while count<100:
			print "#",
			self.__Estep()
			self.__Mstep()

			count += 1

		print "#"
		return 1

if __name__ == "__main__":
	kl = kalman(10,10)
	data = kl.ssm.generate_data(20)
	#kl.set_data(data[1])
	#kl.pfs()
	#for i in range(100): kl.pfs()
	em = Expectation_Maximization(data[0],10)
	em.execute()
	loss = data[1]-em.kl.xs
	#plt.plot(loss)
	plt.plot(em.llh)
	plt.show()
