import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import StateSpaceModel as SSM



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

	def move(self, model):
		self.potition = np.array(model(np.matrix(self.position).T)).T

class PF:
	def __init__(self,
				 data,
				 num_particles=100,
				 dim_particle=10,
				 ):
		
		self.dim_particle = dim_particle
		self.num_particles = num_particles
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

	def __calc_weight(self):
		Yobs = np.matrix(self.obs.T)
		# compare prediction and observation
		for i in range(self.num_particles):
			position = self.particles[i].position
			weights = norm.pdf(x=position, loc=Yobs) # probability density function of normal distribution
			#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)*a2; # pdf * a2
			#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)
			#w[k][j] = w[k-1][j]*gauss((z[k]-Spre[k][j]),mu2,var2)*gauss((Spre[k][j]-Spost[k-1][j]),mu2,var2); # pdf * a2
			#t += w[k][j]; # sum of weights
			#_Neff += w[k][j]*w[k][j]; # threashold for resample

	def __resample(self):
		NP = self.num_particles
		c = [0 for i in range(NP)]
		u = [0 for i in range(NP)]
		for i in range(NP):
			c[i+1] = c[i] + self.particles[i+1].weight
		#u[0] = ((double)random()/RAND_MAX)/(double)jmax;
		u[0] = np.random.randn(1)/NP
		print "Resampling!\n1/jmax=%lf,u[%d][0]=%lf\n",1/NP,k,u[0]
		for j in range(NP):
			i = 0
			u[j] = u[0] + (1.0/NP)*j
			while u[j]>c[i]: i += 1
			#Spost[k][j] = Spre[k][i]
			self.particles[j].position = self.particles[i].position
			#w[k][j] = 1.0/NP
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
				self.particles = [Particle(d=DP, w=1/NP) for i in range(NP)]
			else:
				# move each particles by system equation
				for i in range(NP):
					self.particles[i].move(self.ssm.sys_eq)

			# store prediction distribution
			for i in range(NP):
				x_prediction = self.particles[i].position
				x_prediction.index = [n]
				pd.concat([self.particles[i].x_prdc, x_prediction], axis=0)

			# calculate weight of each particle
			self.__calc_weight()

			# resample to avoid degeneracy problem
			# decide posterior distrobution
			if Neff < NT: # do resampling
			    self.__resample()
			#else: # Posteriors are set to prior 
				#for j in range(NP):
					#self.particles[j].position = self.particles[j].position

			# store filtering distribution
			for i in range(NP):
				x_filtering = self.particles[i].position
				x_filtering.index = [n]
				pd.concat([self.particles[i].x_fltr, x_prediction], axis=0)


if __name__ == "__main__":
	print "particle.py: directly called from main proccess."
