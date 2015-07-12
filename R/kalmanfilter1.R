#data
Q <- 1
R <- 1
x0mean <- 0
x0var <- 1
A <- 1
B <- 1

sys <- function(x){return(A*x + rnorm(1,0,Q))}
obs <- function(x){return(B*x + rnorm(1,0,R))}


X0 <- rnorm(1,x0mean,x0var)
X <- cbind(c(sys(X0)))
Y <- matrix(0)
Y <- Y[-1,]
for(i in 1:100){
  X <- cbind(X,c(sys(X[1,i])))
  Y <- cbind(Y,c(obs(X[1,i])))
}


#
#kalmanfilter
#

sigmaPri <- var(Y[1,])
xPri <- X0
x <- matrix(0)
x <- x[-1,]
for(i in 1:100){
  #filtering
  K <- sigmaPri*B/(B*sigmaPri*B + R)
  x <- cbind(x,xPri + K*(Y[1,i] - B*xPri))
  sigmaPost <- (1 - K*B)*sigmaPri

  #prediction  
  xPri <- A*x[1,i]
  sigmaPri <- A*sigmaPost*A + Q
}

plot(X[1,],xlim=c(0,100),ylim=c(-10,10),type="l")
par(new=T)
plot(x[1,],xlim=c(0,100),ylim=c(-10,10),type="o",col="blue")
