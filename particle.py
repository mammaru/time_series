import numpy as np
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from timeseries import StateSpaceModel as SSM


class Particle(SSM):
	def __init__(self, p, k):
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

	def set_data(self, data):
		self.obs = data
		self.unequal_intarval_flag = True if sum(np.sum(data)) else False
		self.missing_data_flag = True if sum(np.sum(data)) else False


void Particlefilter::filter(FILE *fp)
{
	struct kesseki particle[kaisu][PARTICLE_NUMBER]; 



	/**************************************************

	Begin prediction

	**************************************************/

	/*
	* Scattering initial distribution
	*/
	double tmp_position
		for (j=0;j<PARTICLE_NUMBER;j++) {
			tmp_position[0] = (360/PARTICLE_NUMBER)*j*PI/180;
			tmp_position[1] = (360/PARTICLE_NUMBER)*j*PI/180;
			particle[0][j].position = tmp_position;//均等にばらまく
		}


		/*
		* prediction and resampling
		*/
		//ofstream EyeAngularVelocity("./filtersimulation/eye_angular_velocity.txt");

		for (k=1;k<kaisuu;k++){//time


			/*予測*/
			//それぞれのパーティクルが次の位置に移動
			//つまり事前分布決定
			//そのパーティクルの移動による眼振出力
			for (j=0;j<PARTICLE_NUMBER;j++){
				particle[k][j].nystagmus = bppvSim(k,particle[k-1][j].position);
				//EyeAngularVelocity << output[0] << " " << output[1] << " " << output[2] << endl;

			}

			/*尤度推定*/
			//実データと眼振を比較する
			//実際には眼球の三次元速度か？
			for (j=0;j<PARTICLE_NUMBER;j++){
				particle[k][j].likelihood = calc_likelihood(particle[k][j].nystagmus);
			}


			/*リサンプリング*/
			//パーティクルの尤度に基づいた再配置
			//つまり事後分布決定
			//三次元だけどどうする？
			for(j=0;j<PARTICLE_NUMBER;j++){
				particle[k][j].position = resample(kesseki[k][j].likelihood);
			}





		}

		/*結石位置表示*/


}

double ParticleFilter::calc_likelihood(double *nystagmus)
{
	/*尤度推定*/
	//実データと眼振を比較する
	//実際には眼球の三次元速度か？
	
	//w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)*a2;/*確率密度関数の値×a2*/
	//w[k][j] = gauss((z[k]-Spre[k][j]),mu2,var2)/wgoukei;
	//w[k][j] = w[k-1][j]*gauss((z[k]-Spre[k][j]),mu2,var2)*gauss((Spre[k][j]-Spost[k-1][j]),mu2,var2);/*確率密度関数の値×a2*/

	//t += w[k][j];/*重み合計*/
	//_Neff += w[k][j]*w[k][j];/*リサンプリング判定のため*/



}

double ParticleFilter::resample(double likelihood)//引数は尤度、返り値は結石位置
{
	/*リサンプリング*/
	//パーティクルの次の位置決定
	//三次元だけどどうする？
	if(Neff<NT){
		c[0]=0; 
		for(i=1; i<jmax; i++){
			c[i] = c[i-1] + w[k][i];
		}     
		u[0] = ((double)random()/RAND_MAX)/(double)jmax;
		printf("Resampling!\n1/jmax=%lf,u[%d][0]=%lf\n",(double)1/jmax,k,u[0]);
		for(j=0; j<jmax; j++){
			i = 0;
			u[j] = u[0] + (double)(1.0/jmax)*j;
			while(u[j]>c[i]) i++;
			Spost[k][j] = Spre[k][i];
			w[k][j] = 1.0/jmax;
		}

		if(k==9)for(j=0;j<jmax;j++)printf("%.20lf %.20lf\n",c[j],u[j]);


	}else{/*リサンプリングしない*/
		for(j=0; j<jmax; j++){
			Spost[k][j] = Spre[k][j];
		}
	}

	/*リサンプリング*/
	//for(j=0;j<jmax;++j){
	//ran=(double)random()/RAND_MAX;
	//for(k=0;k<jmax;++k){
	//if(ran<a[k+1] && a[k]<ran){
	//  s=k;
	//}
	//}
	//Spost[k][j]=Spre[k][s];




}
