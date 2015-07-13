import numpy as np
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import StateSpaceModel as ssm
from timeseries import Kalman as kalman

class EM:
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
		self.kl.set_pfs_results()
		
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
	data = kl.ssm.gen_data(N)
	#kl.set_data(data[1])
	#kl.pfs()
	#for i in range(100): kl.pfs()
	em = EM(data[1], sys_k)
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
