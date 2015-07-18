import numpy as np
#import pandas as pd
from scipy.stats import norm
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import StateSpaceModel as SSM



class Particle:
	def __init__(self, d, w=0):
		self.dim = d
		self.position = np.random.randn(dim)
		self.weight = w

	def move(self, model):
		self.potition = model(self.position)
		
class PF:
	def __init__(self, num_particles=100, dim=10, model=SSM.obs_eq):
		self.dim_particle = dim
		self.num_particles = num_particles
		self.particles = [Particle(dim, 1/num_particles) for i in range(num_particles)]
		self.prediction_func = model
	
	def set_data(self, data):
		self.obs = data
		self.obs_dim = data.shape[1]
		self.N = data.shape[0]
		#self.unequal_intarval_flag = True if sum(np.sum(data)) else False
		#self.missing_data_flag = True if sum(np.sum(data)) else False

	def calc_llh(self, Y):
		# compare prediction and observation
		for i in range(self.num_particles):
			position = self.particles[i].position
			weights = norm.pdf(x=position) # probability density function of normal distribution
		#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)*a2; # probability density function * a2
		#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)
		#w[k][j] = w[k-1][j]*gauss((z[k]-Spre[k][j]),mu2,var2)*gauss((Spre[k][j]-Spost[k-1][j]),mu2,var2); # pdf*a2

		#t += w[k][j]; # sum of weights
		#_Neff += w[k][j]*w[k][j]; # threashold for resample

	def resample(self):

		if(Neff<NT): # Do resampling
			c[0]=0; 
			for i in range(jmax):
				c[i] = c[i-1] + w[k][i];
		
			#u[0] = ((double)random()/RAND_MAX)/(double)jmax;
			u[0] = np.random.randn(1)
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
		PA = self.num_particles

		# Scattering initial distribution
		#double tmp_position
		for j in range(PARTICLE_NUMBER):
			tmp_position[0] = (360/PARTICLE_NUMBER)*j*PI/180;
			tmp_position[1] = (360/PARTICLE_NUMBER)*j*PI/180;
			particle[0][j].position = tmp_position; # scattering

		# prediction and resampling		
		for n in range(N):
			
			# prediction to move each particles next position
			# decide prior distribution
			
			#for (j=0;j<PARTICLE_NUMBER;j++){
				#particle[k][j].nystagmus = bppvSim(k,particle[k-1][j].position);
				#//EyeAngularVelocity << output[0] << " " << output[1] << " " << output[2] << endl;
			#}

			# calculate likelihood of each particle
			for j in range(PARTICLE_NUMBER):
				particle[k][j].likelihood = calc_likelihood(particle[k][j].nystagmus);


			# resampling using likelihood
			# decide posterior distrobution
			for j in range(PARTICLE_NUMBER):
				particle[k][j].position = resample(kesseki[k][j].likelihood);




if __name__ == "__main__":
	print "particle.py: directly called from main proccess."
