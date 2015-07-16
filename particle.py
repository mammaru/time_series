import numpy as np
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import StateSpaceModel as SSM


class Particle:
	def __init__(self):
		self.position = 0

	def set_position(self, pos):
		self.position = pos


class Particles:
	def __init__(self, num_particles):
		self.NUM = num_particles
		self.particle = []
		for i in range(num_particles): self.particle[i] = Particle
		
		


class PF(Particles):
	def set_data(self, data):
		self.obs = data
		self.unequal_intarval_flag = True if sum(np.sum(data)) else False
		self.missing_data_flag = True if sum(np.sum(data)) else False

		# constant for particle filter
		self.ssm = SSM(p, k)
		# variable for particle filter
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


	def calc_llh(self):
		# compare prediction and observation
		#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)*a2;/*確率密度関数の値×a2*/
		#w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)/wgoukei;
		#w[k][j] = w[k-1][j]*gauss((z[k]-Spre[k][j]),mu2,var2)*gauss((Spre[k][j]-Spost[k-1][j]),mu2,var2); # pdf*a2

		#t += w[k][j];/*重み合計*/
		#_Neff += w[k][j]*w[k][j];/*リサンプリング判定のため*/

	def resample(self):

		if(Neff<NT):
			c[0]=0; 
			for(i=1; i<jmax; i++){
				c[i] = c[i-1] + w[k][i];
			}     
			u[0] = ((double)random()/RAND_MAX)/(double)jmax;
			printf("Resampling!\n1/jmax=%lf,u[%d][0]=%lf\n",(double)1/jmax,k,u[0]);
			for(j=0; j<jmax; j++){
				i = 0;
				u[j] = u[0] + (double)(1.0/jmax)*j;
				while(u[j]>c[i]): i++;
				Spost[k][j] = Spre[k][i];
				w[k][j] = 1.0/jmax;
			}

			if(k==9)for(j=0;j<jmax;j++): printf("%.20lf %.20lf\n",c[j],u[j]);


		else:
			for(j=0; j<jmax; j++){
				Spost[k][j] = Spre[k][j];
			}

		# exec resampling
		#for(j=0;j<jmax;++j){
			#ran=(double)random()/RAND_MAX;
			#for(k=0;k<jmax;++k){
				#if(ran<a[k+1] && a[k]<ran){
					#s=k;
				#}
			#}
            #Spost[k][j]=Spre[k][s];
		#}

	def filtering(self, num_particles):
		PA = num_particles

		# Scattering initial distribution
		double tmp_position
		for (j=0;j<PARTICLE_NUMBER;j++) {
			tmp_position[0] = (360/PARTICLE_NUMBER)*j*PI/180;
			tmp_position[1] = (360/PARTICLE_NUMBER)*j*PI/180;
			particle[0][j].position = tmp_position; # scattering
		}

		# prediction and resampling		
		for (k=1;k<kaisuu;k++){

			# prediction to move each particles next position
			# decide prior distribution
			for (j=0;j<PARTICLE_NUMBER;j++){
				particle[k][j].nystagmus = bppvSim(k,particle[k-1][j].position);
				//EyeAngularVelocity << output[0] << " " << output[1] << " " << output[2] << endl;
			}

			# calculate likelihood of each particle
			for (j=0;j<PARTICLE_NUMBER;j++){
				particle[k][j].likelihood = calc_likelihood(particle[k][j].nystagmus);
			}


			# resampling using likelihood
			# decide posterior distrobution
			for(j=0;j<PARTICLE_NUMBER;j++){
				particle[k][j].position = resample(kesseki[k][j].likelihood);
			}
		}


