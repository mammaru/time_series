import numpy as np
#import pandas as pd
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

	def move(self, model):
		self.potition = model(self.position)

class PF:
	def __init__(self,
				 num_particles=100,
				 dim_particle=10,
				 sys_eq=SSM.sys_eq,
				 obs_eq=SSM.obs_eq
				 ):
		
		self.dim_particle = dim_particle
		self.num_particles = num_particles
		#self.particles = [Particle(d=dim, w=1/num_particles) for i in range(num_particles)]
		self.sys_eq = sys_eq
		self.obs_eq = obs_eq
	
	def set_data(self, data):
		self.obs = data
		self.obs_dim = data.shape[1]
		self.N = data.shape[0]
		#self.unequal_intarval_flag = True if sum(np.sum(data)) else False
		#self.missing_data_flag = True if sum(np.sum(data)) else False

	def __calc_llh(self, Y):
		# compare prediction and observation
		for i in range(self.num_particles):
			position = self.particles[i].position
			weights = norm.pdf(x=position, loc=self.obs) # probability density function of normal distribution
			#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)*a2; # pdf * a2
			#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)
			#w[k][j] = w[k-1][j]*gauss((z[k]-Spre[k][j]),mu2,var2)*gauss((Spre[k][j]-Spost[k-1][j]),mu2,var2); # pdf * a2
			#t += w[k][j]; # sum of weights
			#_Neff += w[k][j]*w[k][j]; # threashold for resample

	def __resample(self):
		NP = self.num_particles
		if(Neff<NT): # Do resampling
			c = [0 for i in range(NP)]; 
			for i in range(jmax):
				c[i] = c[i-1] + self.particles[i].weight
			#u[0] = ((double)random()/RAND_MAX)/(double)jmax;
			u[0] = np.random.randn(1)/NP
			print "Resampling!\n1/jmax=%lf,u[%d][0]=%lf\n",1/jmax,k,u[0]
			for j in range(jmax):
				i = 0;
				u[j] = u[0] + (double)(1.0/jmax)*j;
				while u[j]>c[i]: i += 1
				Spost[k][j] = Spre[k][i];
				w[k][j] = 1.0/jmax;
			if k==9:
				for j in range(jmax): print "%.20lf %.20lf\n",c[j],u[j]
		else: # Posteriors are set to prior 
			for j in range(jmax):
				Spost[k][j] = Spre[k][j];


	def execute(self):
		DP = self.dim_particle
		NP = self.num_particles
		N = self.N
		
		# Scattering particles for initial distribution
		self.particles = [Particle(d=DP, w=1/NP) for i in range(NP)]

		# prediction and resampling		
		for n in range(N):
			
			# prediction to move each particles next position
			# decide prior distribution
			
			#for (j=0;j<PARTICLE_NUMBER;j++){
				#particle[k][j].nystagmus = bppvSim(k,particle[k-1][j].position);
				#//EyeAngularVelocity << output[0] << " " << output[1] << " " << output[2] << endl;
			#}

			# calculate likelihood of each particle
			for j in range(NP):
				particle[k][j].likelihood = self.__calc_llh(particle[k][j].nystagmus);


			# resample to avoid degeneracy problem
			# decide posterior distrobution
			for j in range(NP):
				particle[k][j].position = self.__resample(kesseki[k][j].likelihood);




if __name__ == "__main__":
	print "particle.py: directly called from main proccess."
