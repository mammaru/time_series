#EM algorithm

#Y1~N(mu1,var1), Y2~N(mu2,var2)
#Y~(1-delta)*Y1 + delta*Y2
mu1 <- 5
sd1 <- sqrt(0.87)
mu2 <- 1
sd2 <- sqrt(0.77)

if(0){
Y <- c()
for(i in 1:20){
	delta <- as.integer(runif(1,min=0,max=2))
	y <- (1-delta)*rnorm(1,mu1,sd1) + delta*rnorm(1,mu2,sd2)
	Y <- c(Y,y)
}
}

Y <- c(-0.39,0.12,0.94,1.67,1.76,2.44,3.72,4.28,4.92,5.53,0.06,0.48,1.01,1.68,1.80,3.25,4.12,4.60,5.28,6.22)

#EM

#initialization
param_new <- list(
	MU1 = Y[as.integer(runif(1,min=1,max=21))],
	SD1 = sqrt(var(Y)),
	MU2 = Y[as.integer(runif(1,min=1,max=21))],
	SD2 = sqrt(var(Y)),
	PI = 0.5
	)
param_old <- param_new
llh <- c(-1000)
count <- 0
flag <- 0

while(!flag){
count <- count+1

#Expectation
gamma <- rep(0,20)
for(i in 1:20){
	tmp1 <- param_old$PI*dnorm(Y[i],param_old$MU2,param_old$SD2)
	tmp2 <- (1-param_old$PI)*dnorm(Y[i],param_old$MU1,param_old$SD1) + param_old$PI*dnorm(Y[i],param_old$MU2,param_old$SD2)
	gamma[i] <- tmp1/tmp2
}

#Maximization
tmp1 <- 0
tmp2 <- 0
for(i in 1:20){
	tmp1 <- tmp1 + (1-gamma[i])*Y[i]
	tmp2 <- tmp2 + (1-gamma[i])
}
param_new$MU1 <- tmp1/tmp2

tmp1 <- 0
tmp2 <- 0
for(i in 1:20){
	tmp1 <- tmp1 + (1-gamma[i])*((Y[i]-param_old$MU1)^2)
	tmp2 <- tmp2 + (1-gamma[i])	
}
param_new$SD1 <- tmp1/tmp2

tmp1 <- 0
tmp2 <- 0
for(i in 1:20){
	tmp1 <- tmp1 + gamma[i]*Y[i]
	tmp2 <- tmp2 + gamma[i]#sum(gamma)
}
param_new$MU2 <- tmp1/tmp2

tmp1 <- 0
tmp2 <- 0
for(i in 1:20){
	tmp1 <- tmp1 + gamma[i]*((Y[i]-param_old$MU2)^2)
	tmp2 <- tmp2 + gamma[i]#sum(gamma)
}
param_new$SD2 <- tmp1/tmp2

tmp <- 0
for(i in 1:20)tmp<-tmp+gamma[i]
param_new$PI <- tmp/20

#likelihood
tmp1 <- 0
tmp2 <- 0
tmp3 <- 0
tmp4 <- 0
for(i in 1:20){
	tmp1 <- tmp1 + (1-gamma[i])*log(dnorm(Y[i],param_new$MU1,param_new$SD1))+gamma[i]*log(dnorm(Y[i],param_new$MU2,param_new$SD2))
	tmp2 <- tmp2 + (1-gamma[i])*log(1-param_new$PI)+gamma[i]*log(param_new$PI)
	#tmp3 <- tmp3 + log((1-param_new$PI)*dnorm(Y[i],param_new$MU1,param_new$SD1)+param_new$PI*dnorm(Y[i],param_new$MU2,param_new$SD2))
	#tmp4 <- tmp4 + log((1-param_new$PI)*dnorm(Y[i],param_new$MU1,param_new$SD1)+param_new$PI*dnorm(Y[i],param_new$MU2,param_new$SD2))
}
llh <- c(llh,tmp1+tmp2)
#llh <- c(llh,tmp3)
#llh <- c(llh,tmp4)

#check
diff <- llh[count+1] - llh[count]
if((diff>0&&diff<0.0001)||count==50){
	flag<-1
	}else{
		param_old <- param_new
		}

}


#plot
param <- param_old
g <- function(y,param){
	d <- (1-param$PI)*dnorm(y,param$MU1,param$SD1) + param$PI*dnorm(y,param$MU2,param$SD2)
	return(d)
}
hist(Y,breaks=14,freq=FALSE,xlim=c(min(Y),max(Y)),ylim=c(0,1),col="red")
lines(-1:10,g(-1:10,param),xlim=c(min(Y),max(Y)),ylim=c(0,1))
#lines(gamma,xlim=c(-1,10),ylim=c(0,1))

plot(llh,type="l")



