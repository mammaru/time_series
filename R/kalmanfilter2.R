library(MASS)
#library(igraph)

#
#data
#

if(1){
T <- 200
k <- 1 #state
m <- 1 #state noise
p <- 1 #observation
#Q <- matrix(rnorm(k*k,0,10),k,k)
Q <- diag(1,m)
#R <- matrix(c(3,1,2,1,5,4,2,4,2),p,p)
R <- diag(10,p)
X0mean <- rep(0,k)
X0var <- diag(k)
#F <- matrix(runif(k*k,min=-1,max=1),k,k)
#F <- matrix(c(1.03,0,0,0,0,1,0,0,0,0,0.8,0,0.5,0,0,-0.3),k,k)
#F <- matrix(c(2,1,0,0,0,-1,0,0,0,0,0,0,-1,1,0,0,0,-1,0,1,0,0,-1,0,0),k,k) #季節調整kmp521
F <- matrix(c(1),k,k) #トレンドモデル1次元kmp111
#F <- matrix(c(2,1,-1,0),k,k) #トレンドモデル2次元kmp211
#G <- diag(k*m)
#G <- matrix(c(1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0,1,0),k,m)
#G <- matrix(c(1,0,0,0,0,0,0,1,0,0),k,m) #季節調整kmp521
G <- matrix(c(1),k,m) #トレンドモデル1次元kmp111
#G <- matrix(c(1,0),k,m) #トレンドモデル2次元kmp211
#H <- matrix(rnorm(p*k,0,0.5),p,k)
#H <- cbind(c(rep(1,250),rep(0,750)),c(rep(0,250),rep(1,250),rep(0,500)),c(rep(0,500),rep(1,250),rep(0,250)),c(rep(0,750),rep(1,250)))
#H <- diag(1,p)
#H <- matrix(c(1,0,1,0,0),p,k) #季節調整kmp521
H <- matrix(c(1),p,k) #トレンドモデル1次元kmp111
#H <- matrix(c(1,0),p,k) #トレンドモデル2次元kmp211

#tkplot(graph.data.flame(A))

sys <- function(x){
  v <- mvrnorm(1,rep(0,m),Q)
  return(F%*%x + G%*%v)
}
obs <- function(x){
  w <- mvrnorm(1,rep(0,p),R)
  return(H%*%x + w)
}



X0 <- mvrnorm(1,X0mean,X0var)
X <- sys(X0)
Y <- matrix(rep(0,p),p,1)
Y <- Y[,-1]

for(i in 1:T){
  X <- cbind(X,c(sys(X[,i])))
  Y <- cbind(Y,c(obs(X[,i])))
}

}

if(1){
#
#kalmanfilter
#
Q <- diag(1,m)
R <- diag(10,p)

xPri <- sys(X0)
sigmaPri <- list((X[,1]-xPri[,1])%*%t(X[,1]-xPri[,1]))
xPost <- matrix(rep(0,k),k,1)
xPost <- xPost[,-1]
sigmaPost <- as.list(NULL)
for(i in 1:T){
  #filtering
  K <- sigmaPri[[i]]%*%t(H)%*%solve(H%*%sigmaPri[[i]]%*%t(H) + R)
  xPost <- cbind(xPost,xPri[,i] + K%*%(Y[,i] - H%*%xPri[,i]))
  sigmaPost[[i]] <- sigmaPri[[i]] - K%*%H%*%sigmaPri[[i]]

  #prediction  
  xPri <- cbind(xPri,F%*%xPost[,i])
  sigmaPri[[i+1]] <- F%*%sigmaPost[[i]]%*%t(F) + G%*%Q%*%t(G)
}


#
#smoother
#
xT <- matrix(xPost[,T],k,1)
sigmaT <- as.list(NULL)
sigmaT[[T]] <- sigmaPost[[T]]
for(i in T:2){
  J <- sigmaPost[[i-1]]%*%t(F)%*%solve(sigmaPri[[i]])
  xT <- cbind(xPost[,i-1] + J%*%(xT[,1]-xPri[,i]),xT)
  sigmaT[[i-1]] <- sigmaPost[[i-1]] + J%*%(sigmaT[[i]]-sigmaPri[[i]])%*%t(J)
}


#
#estimating observation
#
y <- matrix(rep(0,p),p,1)
y <- y[,-1]
yPri <- y
yPost <- y
yT <- y
dPri <- as.list(NULL)
dPost <- as.list(NULL)
dT <- as.list(NULL)

for(i in 1:T){
  yPri <- cbind(yPri,c(H%*%xPri[,i]))
  yPost <- cbind(yPost,c(H%*%xPost[,i]))
  yT <- cbind(yT,c(H%*%xT[,i]))
  dPri[[i]] <- H%*%sigmaPri[[i]]%*%t(H) + R
  dPost[[i]] <- H%*%sigmaPost[[i]]%*%t(H) + R
  dT[[i]] <- H%*%sigmaT[[i]]%*%t(H) + R 
}

}
  
#
#plot
#
if(k==1){X <- t(matrix(X[,-(T+1)]))}else{X <- X[,-(T+1)]}

for(i in 1:k){

x <- xPri
s <- sigmaPri
plot(X[i,],xlim=c(0,T),ylim=c(min(X[i,],xPri[i,],xPost[i,],xT[i,]),max(X[i,],xPri[i,],xPost[i,],xT[i,])),type="p")
par(new=T)
plot(x[i,],xlim=c(0,T),ylim=c(min(X[i,],xPri[i,],xPost[i,],xT[i,]),max(X[i,],xPri[i,],xPost[i,],xT[i,])),type="l",col="blue")
for(j in 1:T){
  arrows(j, x[i,j] + 2*sqrt(s[[j]][i,i]),
         j, x[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("prediction")

x <- xPost
s <- sigmaPost
plot(X[i,],xlim=c(0,T),ylim=c(min(X[i,],xPri[i,],xPost[i,],xT[i,]),max(X[i,],xPri[i,],xPost[i,],xT[i,])),type="p")
par(new=T)
plot(x[i,],xlim=c(0,T),ylim=c(min(X[i,],xPri[i,],xPost[i,],xT[i,]),max(X[i,],xPri[i,],xPost[i,],xT[i,])),type="l",col="blue")
for(j in 1:T){
  arrows(j, x[i,j] + 2*sqrt(s[[j]][i,i]),
         j, x[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("filtering")

x <- xT
s <- sigmaT
plot(X[i,],xlim=c(0,T),ylim=c(min(X[i,],xPri[i,],xPost[i,],xT[i,]),max(X[i,],xPri[i,],xPost[i,],xT[i,])),type="p")
par(new=T)
plot(x[i,],xlim=c(0,T),ylim=c(min(X[i,],xPri[i,],xPost[i,],xT[i,]),max(X[i,],xPri[i,],xPost[i,],xT[i,])),type="l",col="blue")
for(j in 1:T){
  arrows(j, x[i,j] + 2*sqrt(s[[j]][i,i]),
         j, x[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("smoothing")

}



if(1){


for(i in 1:p){
y <- yPri
s <- sigmaPri
plot(Y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",lty=2)
par(new=T)
plot(y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",col="blue")
for(j in 1:T){
  arrows(j, x[i,j] + 2*sqrt(s[[j]][i,i]),
         j, x[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("prediction obs")

y <- yPost
s <- sigmaPost
plot(Y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",lty=2)
par(new=T)
plot(y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",col="blue")
for(j in 1:T){
  arrows(j, x[i,j] + 2*sqrt(s[[j]][i,i]),
         j, x[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("filtering obs")

y <- yT
s <- sigmaT
plot(Y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",lty=2)
par(new=T)
plot(y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",col="blue")
for(j in 1:T){
  arrows(j, x[i,j] + 2*sqrt(s[[j]][i,i]),
         j, x[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("smoothing obs")

}

}
