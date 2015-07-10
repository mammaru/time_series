import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from matplotlib import pyplot as plt


class ssm:
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
	
	def generate_data(self, N):
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

class kalman(ssm):
	def __init__(self, p, k):
		# constant for kalman
		self.ssm = ssm(p, k)
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

		x0 = np.matrix(np.random.multivariate_normal(x0mean.T.tolist()[0], np.asarray(x0var))).T
		xp = np.matrix(self.ssm.sys_eq(x0,F,Q))
		vp.append(F*x0var*F.T+Q)

		for i in range(N):
			# filtering
			K = vp[i]*H.T*(H*vp[i]*H.T+R).I
			#print xf
			#print xp[:,i]+K*(Yobs[:,i]-H*xp[:,i])
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
		

class Expectation_Maximization:
	def __init__(self, data, *sys_dim):
		# kalman instance for em
		self.data = data
		self.p = data.shape[1]
		self.k = sys_dim[0] if sys_dim else self.p
		self.N = data.shape[0]
		
		self.kl = kalman(self.p, self.k)
		self.kl.set_data(self.data)
		# variable for em
		self.x0mean = self.kl.x0mean
		self.x0var = self.kl.x0var
		self.F = self.kl.F
		self.Q = self.kl.Q
		self.H = self.kl.H
		self.R = self.kl.R
		self.llh = []
		
	def __Estep(self):
		""" Private method: Expectation step of EM algorithm for SSM """

		# execute kalman's algorithm(prediction, filtering and smoothing)
		#Yobs = np.matrix(self.data.T)
		self.kl.F = self.F
		self.kl.H = self.H
		self.kl.Q = self.Q
		self.kl.R = self.R
		self.kl.x0mean = self.x0mean
		self.kl.x0var = self.x0var
		self.kl.pfs()

		p = self.p
		k = self.k
		N = self.N
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

		#likelihood = (-1/2)*(np.log(np.linalg.det(x0var)) + np.trace(x0var.I*(vs0+(xs0-x0mean)*(xs0-x0mean).T)) + N*np.log(np.linalg.det(Q)) + np.trace(Q.I*(S11-S10*F.T-F*S10.T+F*S00*F.T)) + N*np.log(np.linalg.det(R)) + np.trace(R.I*tmp) + (k+N*(k+p))*np.log(2*np.pi))

		logllh = np.log(np.linalg.det(x0var)) + np.trace(np.linalg.inv(x0var)*(vs0+(xs0-x0mean)*(xs0-x0mean).T)) + N*np.log(np.linalg.det(R)) + np.trace(np.linalg.inv(R)*(Syy+H*S11*H.T-Syx*H.T-H*Syx.T)) + N*np.log(np.linalg.det(Q)) + np.trace(np.linalg.inv(Q)*(S11+F*S00*F.T-S10*F.T-F*S10.T)) + (k+N*(k+p))*np.log(2*np.pi)

		print "\t(1)\t", np.trace(np.linalg.inv(x0var)*(vs0+(xs0-x0mean)*(xs0-x0mean).T))
		print "\t(2)\t", N*np.log(np.linalg.det(R))
		print "\t(3)\t", np.trace(np.linalg.inv(R)*(Syy+H*S11*H.T-Syx*H.T-H*Syx.T))
		print "\t(4)\t", N*np.log(np.linalg.det(Q))
		print "\t(5)\t", np.trace(np.linalg.inv(Q)*(S11+F*S00*F.T-S10*F.T-F*S10.T))

		logllh = (-1/2)*logllh
		#print logllh
		self.llh.append(logllh)


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
		vs0 = self.kl.vs0
		self.F = S10*S00.I
		self.H = Syx*S11.I
		#self.Q = (S11 - S10*S00.I*S10.T)/N
		self.R = np.diag(np.diag(Syy - Syx*np.linalg.inv(S11)*Syx.T))/N
		self.x0mean = np.asarray(xs0.T)
		self.x0var = vs0

 
	def execute(self):
		""" Execute EM algorithm """
		count = 0
		diff = 100
		while diff>1e-3 and count<5000:
			print count,
			self.__Estep()
			self.__Mstep()
			if count>0: diff = abs(self.llh[count] - self.llh[count-1])
			print "\tllh:", self.llh[count]
			count += 1
		#print "#"
		#return 1

if __name__ == "__main__":
	obs_p = 2
	sys_k = 2
	N = 20
	kl = kalman(obs_p,sys_k)
	data = kl.ssm.generate_data(N)
	#kl.set_data(data[1])
	#kl.pfs()
	#for i in range(100): kl.pfs()
	em = Expectation_Maximization(data[1], sys_k)
	em.execute()

	if 1:
		fig, axes = plt.subplots(np.int(np.ceil(sys_k/3.0)), 3, sharex=True)
		j = 0
		for i in range(3):
			while j<sys_k:
				if sys_k<=3:
					axes[j%3].plot(data[0][j], "k--", label="obs")
					axes[j%3].plot(em.kl.xp[j], label="prd")
					axes[j%3].legend(loc="best")
					axes[j%3].set_title(j)
				else:
					axes[i, j%3].plot(data[0][j], "k--", label="obs")
					axes[i, j%3].plot(em.kl.xp[j], label="prd")
					axes[i, j%3].legend(loc="best")
					axes[i, j%3].set_title(j)
				j += 1
				if j%3 == 2: break

		fig.show()
		
	#loss = data[0]-em.kl.xs
	#plt.plot(loss)
	#plt.plot(em.llh)
	#plt.show()
