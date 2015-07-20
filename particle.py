import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import StateSpaceModel as SSM

Nthr = 1e-1

class Particle:
	def __init__(self, d, w=0):
		try:
			self.dim = d
		except:
			print "Dimention of particle must be specified as augument \"d\"."
		self.position = np.random.randn(d)
		self.weight = w
		self.prd = DataFrame(np.empty([0, d]))
		self.flt = DataFrame(np.empty([0, d]))
		self.smt = DataFrame(np.empty([0, d]))

	def move(self, model): # model = some equation such like system equation.
		self.position = np.array(model(np.matrix(self.position).T)).T

class PF:


	def __init__(self,
				 data,
				 num_particles=100,
				 dim_particle=10,
				 ):
		
		self.dim_particle = dim_particle
		self.num_particles = num_particles
		self.Neff = 0
		#self.particles = [Particle(d=dim, w=1/num_particles) for i in range(num_particles)]

		# constant for kalman
		self.obs = data
		self.obs_dim = data.shape[1]
		self.N = data.shape[0]

		self.ssm = SSM(self.obs_dim, self.dim_particle)
	
	def set_data(self, data):
		self.obs = data
		self.obs_dim = data.shape[1]
		self.N = data.shape[0]
		#self.unequal_intarval_flag = True if sum(np.sum(data)) else False
		#self.missing_data_flag = True if sum(np.sum(data)) else False

	def __calc_weight(self, n):
		Yobs = self.obs
		obs = np.array(Yobs.ix[n])
		NP = self.num_particles
		sum_w = 0
		self.Neff = 0
		
		for i in range(NP):
			#print i
			prd = np.array(self.ssm.obs_eq(x=np.matrix(self.particles[i].position).T))
			# compare prediction and observation
			distance = np.sqrt(np.sum((obs - prd)**2))
			#print distance
			 # probability density function of normal distribution
			self.particles[i].weight = norm.pdf(x=distance)
			#self.particles[i].weight = 1.0/NP
			
			sum_w += self.particles[i].weight
			#self.Neff += (self.particles[i].weight)**2
			#t += w[k][j]; # sum of weights
			#_Neff += w[k][j]*w[k][j]; # threashold for resample

		for i in range(NP): self.particles[i].weight /= sum_w
		self.Neff = np.sum([(self.particles[i].weight)**2 for i in range(NP)])

	def __resample(self):
		NP = self.num_particles
		c = [0 for i in range(NP)]
		u = [0 for i in range(NP)]
		for i in range(NP)[1:]:
			c[i] = c[i-1] + self.particles[i].weight
		#u[0] = ((double)random()/RAND_MAX)/(double)jmax;
		u[0] = np.random.randn(1)[0]/NP
		#print "Resampling!\n1/jmax=%lf,u[%d][0]=%lf\n",1/NP,k,u[0]
		for j in range(NP):
			i = 0
			u[j] = u[0] + (1.0/NP)*j
			while u[j]>c[i] and j > i:
				#print i, j
				i += 1

			self.particles[j].position = self.particles[i].position
			self.particles[j].weight = 1.0/NP

	def execute(self):
		DP = self.dim_particle
		NP = self.num_particles
		N = self.N
		
		# prediction and resampling		
		for n in range(N):
			
			# prediction to move each particles next position
			# decide prior distribution
			if n==0:
				# Scattering particles for initial distribution
				self.particles = [Particle(d=DP, w=1.0/NP) for i in range(NP)]
			else:
				# move each particles by system equation
				for i in range(NP):
					self.particles[i].move(self.ssm.sys_eq)

			# store prediction distribution
			for i in range(NP):
				x_prediction = DataFrame(self.particles[i].position).T
				x_prediction.index = [n]
				self.particles[i].prd = pd.concat([self.particles[i].prd, x_prediction], axis=0)

			# calculate weight of each particle
			self.__calc_weight(n)

			# resample to avoid degeneracy problem
			# decide posterior distrobution
			print self.Neff
			if self.Neff < Nthr: # do resampling
			    self.__resample()

			# store filtering distribution
			for i in range(NP):
				x_filtering = DataFrame(self.particles[i].position).T
				x_filtering.index = [n]
				self.particles[i].flt = pd.concat([self.particles[i].flt, x_filtering], axis=0)


if __name__ == "__main__":
	print "particle.py: called in main proccess."
	NP = 5
	DP = 1
	N = 50
	
	ssm = SSM(10,DP)
	data = ssm.gen_data(N)

	p = PF(data[1], num_particles=NP, dim_particle=DP)
	p.execute()

	tmp = []
	for i in range(NP):
		tmp_inner = []
		for j in range(N):
			tmp_inner.append(np.mean(p.particles[i].flt.ix[j]))
		tmp.append(tmp_inner)
	tmp = np.array(tmp)
	estimated_sys = DataFrame(np.mean(tmp, axis=0))

	if 1:
		plt.plot(estimated_sys)
		plt.plot(data[0])
		plt.show()
