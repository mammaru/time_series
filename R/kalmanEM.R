library(MASS)
#library(igraph)





#########################################################################################
#data
#########################################################################################

if(1){
T <- 100
k <- 2 #state
m <- 2 #state noise
p <- 3 #observation
#Q <- matrix(rnorm(k*k,0,10),k,k)
Q <- diag(1,m)
#R <- matrix(c(3,1,2,1,5,4,2,4,2),p,p)
Rvar <- 100
R <- diag(Rvar,p)
X0mean <- c(rep(0,k))
X0var <- diag(100,k)

#システム遷移行列F (k,k)
#F <- matrix(runif(k*k,min=-1,max=1),k,k)#とりあえず
#F <- matrix(c(1.2,0,0.3,-1.5),k,k)#k=2
#F <- diag(k)
#F <- matrix(c(1.03,0,0,0,0,1,0,0,0,0,0.8,0,0.5,0,0,-0.3),k,k)
#F <- matrix(c(2,1,0,0,0,-1,0,0,0,0,0,0,-1,1,0,0,0,-1,0,1,0,0,-1,0,0),k,k) #季節調整kmp521
#F <- matrix(c(rep(1,k*k)),k,k)
#F <- matrix(1) #トレンドモデル1次kmp111
F <- matrix(c(2,1,-1,0),k,k) #トレンドモデル2次kmp211

#システムノイズにかかる行列G (k,m) #使わない12/2/19
#G <- diag(k)#要k=mでGなし
#G <- matrix(c(1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0,1,0),k,m)
#G <- matrix(c(1,0,0,0,0,0,0,1,0,0),k,m) #季節調整kmp521
#G <- matrix(c(rep(1,k*m)),k,m)
#G <- matrix(1) #トレンドモデル1次kmp111
#G <- matrix(c(1,0),k,m) #トレンドモデル2次kmp211

#観測行列H (p,k)
#H <- matrix(rnorm(p*k,0,0.5),p,k)
H <- matrix(c(-1,0.2,-2,-1.4,2.4,3.8),p,k)#p=3,k=2
#H <- cbind(c(rep(1,250),rep(0,750)),c(rep(0,250),rep(1,250),rep(0,500)),c(rep(0,500),rep(1,250),rep(0,250)),c(rep(0,750),rep(1,250)))
#H <- diag(1,p)
#H <- matrix(c(1,0,1,0,0),p,k) #季節調整kmp521
#H <- matrix(c(rep(1,p*k)),p,k)
#H <- matrix(1) #トレンドモデル1次kmp111
#H <- matrix(c(1,0),p,k) #トレンドモデル2次kmp211

#tkplot(graph.data.flame(A))

sys <- function(XX,FF){
  #print(F)
  v <- mvrnorm(1,rep(0,m),Q)
  #return(FF%*%XX + GG%*%v)
  return(FF%*%XX + v)
  
}
obs <- function(XX,HH){
  w <- mvrnorm(1,rep(0,p),R)
  return(HH%*%XX + w)
}



X0 <- mvrnorm(1,X0mean,X0var)
X <- sys(X0,F)
Y <- matrix(rep(0,p),p,1)
Y <- Y[,-1]

for(i in 1:T){
  X <- cbind(X,c(sys(X[,i],F)))
  Y <- cbind(Y,c(obs(X[,i],H)))
}

}












#########################################################################################
#EM (F,Q,R,X0mean)
#########################################################################################

if(1){

#初期値
q <- diag(1,m)#推定
r <- diag(1,p)#推定
f <- diag(1,k)#推定
#g <- matrix(c(rep(1,k*m)),k,m)#推定しない
#g <- diag(k)
#h <- matrix(c(rep(1,p*k)),p,k)#推定しない
h <- H
x0mean <- rep(0,k)#推定

x0var <- diag(100,k)#これは固定

count <- 0
likelihood <- c(0,1000)
flag <- 0
while(flag==0){

#E
x0 <- mvrnorm(1,x0mean,x0var)
xPri <- sys(x0,f)
#sigmaPri <- list((X[,1]-xPri[,1])%*%t(X[,1]-xPri[,1]))
#sigmaPri <- list(f%*%x0var%*%t(f) + g%*%q%*%t(g))
sigmaPri <- list(f%*%x0var%*%t(f) + q)
xPost <- matrix(rep(0,k),k,1)
xPost <- xPost[,-1]
sigmaPost <- as.list(NULL)
for(i in 1:T){
  #filtering
  K <- sigmaPri[[i]]%*%t(h)%*%solve(h%*%sigmaPri[[i]]%*%t(h) + r)
  xPost <- cbind(xPost,xPri[,i] + K%*%(Y[,i] - h%*%xPri[,i]))
  sigmaPost[[i]] <- sigmaPri[[i]] - K%*%h%*%sigmaPri[[i]]

  #prediction  
  xPri <- cbind(xPri,f%*%xPost[,i])
  #sigmaPri[[i+1]] <- f%*%sigmaPost[[i]]%*%t(f) + g%*%q%*%t(g)
  sigmaPri[[i+1]] <- f%*%sigmaPost[[i]]%*%t(f) + q

}

#尤度
sum1 <- 0
sum2 <- 0
for(i in 1:T){
innovation <- Y[,i] - h%*%xPri[,i]
vari <- h%*%sigmaPri[[i]]%*%t(h) + r
sum1 <- sum1 + log(det(vari))
sum2 <- sum2 + t(innovation)%*%solve(vari)%*%innovation
}
likelihood[count+2] <- (sum1 + sum2)/2

#smoothing
#J <- c(rep(0,T))
J <- as.list(NULL)
J[[T]] <- matrix(rep(0,k*k),k,k)
xT <- matrix(xPost[,T],k,1)
sigmaT <- as.list(NULL)
sigmaT[[T]] <- sigmaPost[[T]]
sigmaLag <- as.list(NULL)
sigmaLag[[T]] <- f%*%sigmaPost[[T-1]] - K%*%h%*%sigmaPost[[T-1]]

for(i in T:2){
  J[[i-1]] <- sigmaPost[[i-1]]%*%t(f)%*%solve(sigmaPri[[i]])
  xT <- cbind(xPost[,i-1] + J[[i-1]]%*%(xT[,1]-xPri[,i]),xT)
  sigmaT[[i-1]] <- sigmaPost[[i-1]] + J[[i-1]]%*%(sigmaT[[i]]-sigmaPri[[i]])%*%t(J[[i-1]])
}
for(i in T:3){
  sigmaLag[[i-1]] <- sigmaPost[[i-1]]%*%t(J[[i-1]]) + J[[i-1]]%*%(sigmaLag[[i]]-f%*%sigmaPost[[i-1]])%*%t(J[[i-2]])
}
J0 <- x0var%*%t(f)%*%solve(sigmaPri[[1]])
sigmaLag1 <- sigmaPost[[1]]%*%t(J0) + J[[1]]%*%(sigmaLag[[2]]-f%*%sigmaPost[[1]])%*%t(J0)
xT0 <- x0mean + J0%*%(xT[,1]-xPri[,1])
sigmaT0 <- x0var + J0%*%(sigmaT[[1]]-sigmaPri[[1]])%*%t(J0)
#print(c(f%*%x0mean,xPri[,1]))
#print(sigmaT0)
#print(xT0)

#M
S11 <- xT[,1]%*%t(xT[,1]) + sigmaT[[1]]
S10 <- xT[,1]%*%t(xT0) + sigmaLag1
S00 <- xT0%*%t(xT0) + x0var
for(i in 2:T){
  S11 <- S11 + xT[,i]%*%t(xT[,i]) + sigmaT[[i]]
  S10 <- S10 + xT[,i]%*%t(xT[,i-1]) + sigmaLag[[i]]
  S00 <- S00 + xT[,i-1]%*%t(xT[,i-1]) + sigmaT[[i-1]]
}

f <- S10%*%solve(S00)
#q <- (S11 - S10%*%solve(S00)%*%t(S10))/T

r <- 0
for(i in 1:T){
  r <- r + (Y[,i] - h%*%xT[,i])%*%t(Y[,i] - h%*%xT[,i]) + h%*%sigmaT[[i]]%*%t(h)
}
r <- r/T

x0mean <- xT0
#x0var <- sigmaT0

if(abs(likelihood[count+2]-likelihood[count+1]) < 0.00001 || count==100){flag <- 1}#尤度の差が小さければおわり

count <- count + 1

}#whileループ終わり


print(count)
#print(likelihood)
plot(likelihood[-1],type="l")
title("likelihood")
print("F")
print(F)
print(f)
print("-----------------")
print("Q")
print(Q)
print(q)
print("-----------------")
print("R")
print(R)
print(r)
print("-----------------")
print("X0mean")
print(X0mean)
print(x0mean)
#print(c("X0var",X0var,x0var))

}












#########################################################################################
#評価とプロット
#########################################################################################

if(1){

#推定されたパラメータ(x0mean,f,q,r)でカルマンフィルタ実行
x0 <- mvrnorm(1,x0mean,x0var)
xPri <- sys(x0,f)
#sigmaPri <- list((X[,1]-xPri[,1])%*%t(X[,1]-xPri[,1]))
sigmaPri <- list(f%*%x0var%*%t(f) + q)
xPost <- matrix(rep(0,k),k,1)
xPost <- xPost[,-1]
sigmaPost <- as.list(NULL)
for(i in 1:T){
  #filtering
  K <- sigmaPri[[i]]%*%t(h)%*%solve(h%*%sigmaPri[[i]]%*%t(h) + r)
  xPost <- cbind(xPost,xPri[,i] + K%*%(Y[,i] - h%*%xPri[,i]))
  sigmaPost[[i]] <- sigmaPri[[i]] - K%*%h%*%sigmaPri[[i]]
  #prediction  
  xPri <- cbind(xPri,f%*%xPost[,i])
  sigmaPri[[i+1]] <- f%*%sigmaPost[[i]]%*%t(f) + q

}
#smoothing
#J <- c(rep(0,T))
J <- as.list(NULL)
J[[T]] <- matrix(rep(0,k*k),k,k)
xT <- matrix(xPost[,T],k,1)
sigmaT <- as.list(NULL)
sigmaT[[T]] <- sigmaPost[[T]]
for(i in T:2){
  J[[i-1]] <- sigmaPost[[i-1]]%*%t(f)%*%solve(sigmaPri[[i]])
  xT <- cbind(xPost[,i-1] + J[[i-1]]%*%(xT[,1]-xPri[,i]),xT)
  sigmaT[[i-1]] <- sigmaPost[[i-1]] + J[[i-1]]%*%(sigmaT[[i]]-sigmaPri[[i]])%*%t(J[[i-1]])
}
J0 <- x0var%*%t(f)%*%solve(sigmaPri[[1]])
xT0 <- x0mean + J0%*%(xT[,1]-xPri[,1])
sigmaT0 <- x0var + J0%*%(sigmaT[[1]]-sigmaPri[[1]])%*%t(J0)
#print(c(f%*%x0mean,xPri[,1]))
#print(sigmaT0)

#estimating observation
y <- matrix(rep(0,p),p,1)
y <- y[,-1]
yPri <- y
yPost <- y
yT <- y
dPri <- as.list(NULL)
dPost <- as.list(NULL)
dT <- as.list(NULL)
for(i in 1:T){
  yPri <- cbind(yPri,c(h%*%xPri[,i]))
  yPost <- cbind(yPost,c(h%*%xPost[,i]))
  yT <- cbind(yT,c(h%*%xT[,i]))
  dPri[[i]] <- h%*%sigmaPri[[i]]%*%t(h) + r
  dPost[[i]] <- h%*%sigmaPost[[i]]%*%t(h) + r
  dT[[i]] <- h%*%sigmaT[[i]]%*%t(h) + r
}

#MSE
y <- yT
MSE <- (sum((Y-y)^2))/T
MSE
#MSE <- matrix(rep(0,2),1,2)
#MSE <- MSE[-1,]
#MSE <- rbind(MSE,c(100*var/Rvar,(sum((Y-yhat)^2))/T))
#plot(MSE,type="l",xlab="sigma^2(%)",ylab="MSE")

#Yとyプロット
for(i in 1:p){
if(0){
y <- yPri
s <- dPri
plot(Y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l")
par(new=T)
plot(y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",col="blue",lwd=3)
for(j in 1:T){
  arrows(j, y[i,j] + 2*sqrt(s[[j]][i,i]),
         j, y[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("prediction obs")
}
if(0){
y <- yPost
s <- dPost
plot(Y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l")
par(new=T)
plot(y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",col="blue",lwd=3)
for(j in 1:T){
  arrows(j, y[i,j] + 2*sqrt(s[[j]][i,i]),
         j, y[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("filtering obs")
}
if(1){
y <- yT
s <- dT
plot(Y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l")
par(new=T)
plot(y[i,],xlim=c(0,T),ylim=c(min(y[i,],Y[i,]),max(y[i,],Y[i,])),type="l",col="blue",lwd=3)
for(j in 1:T){
  arrows(j, y[i,j] + 2*sqrt(s[[j]][i,i]),
         j, y[i,j] - 2*sqrt(s[[j]][i,i]),
         col = "cyan",
         lwd = 1, lty = 5, length = 0.05, angle = 90, code = 3)
}
title("smoothing obs")
}
}

#EM結果
print(count)
print("F")
print(F)
print(f)
#print("-----------------")
#print("G")
#print(G)
#print(g)
print("-----------------")
print("H")
print(H)
print(h)
print("-----------------")
print("Q")
print(Q)
print(q)
print("-----------------")
print("R")
print(R)
print(r)
print("-----------------")
print("X0mean")
print(X0mean)
print(x0mean)
#print(c("X0var",X0var,x0var))

#MSE
MSE

}



#plot(X[1,],type="l")
#par(new=T)
#plot(Y[1,],type="l")


