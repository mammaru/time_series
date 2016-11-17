#coding: utf-8
"""
Kalman's algorithm(prediction, filtering, smoothing)

Author: mammaru <mauma1989@gmail.com>

"""
import numpy as np
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from .base import SSMBase


class StateSpaceModel(SSMBase):
    def _predict(self, values):
        return {
            "xp": self.F*values["xf"],
            "vp": self.F*values["vf"]*self.F.T+self.Q
            }

    def _filter(self, values):
        K = values["vp"]*self.H.T*(self.H*values["vp"]*self.H.T+self.R).I
        return {
            "xf": values["xp"]+K*(Yobs[:,i]-self.H*values["xp"]),
            "vf": values["vp"]-K*self.H*values["vp"],
            "K": K
            }

    def _smooth(self, values):
        J = values["vf-1"]*self.F.T*values["vp"].I
        return {
            "xs": values["xf-1"]+J*(values["xs"]-values["xp"]),
            "vs": values["vf-2"]+J*(values["vs"]-values["vp"])*J.T,
            "J": J
            }

    def __pfs(self, data):
        """ body of the kalman's method - prediction, filtering and smoothing """
        N = data.shape[0] # number of time points
        Yobs = np.matrix(data.T)
        xp = np.matrix(self.F*self.x0mean)
        vp.append(self.F*self.x0var*self.F.T+self.Q)
        for i in range(N):
            # filtering
            values = self._filter({"xp": xp, "vp": vp})
            xf = xp[:,i]+K*(Yobs[:,i]-self.H*xp[:,i]) if i == 0 else np.hstack([xf, values["xf"]])
            vf.append(values["vf"])
            K = values["K"]
            # prediction
            values = self._predict(values)
            xp = np.hstack([xp, values["xp"]])
            vp.append(values["vp"])

        # smoothing
        J = [np.matrix(np.zeros([k,k]))]
        xs = xf[:,N-1]
        vs.insert(0, vf[N-1])
        vLag.insert(0, self.F*vf[N-2]-K*self.H*vf[N-2])
        for i in reversed(range(N)[1:]):
            values = self._smooth({"xp": xp[:,i], "xf-1": xf[:,i-1], "xs": xs[:,0], "vp": vp[i], "vf-1": vf[i-1], "vf-2": vf[i-2], "vs": vs[0]})
            J.insert(0, values["J"])
            xs = np.hstack([values["xs"], xs])
            vs.insert(0, values["vs"])
        
        for i in reversed(range(N)[2:]):
            vLag.insert(0, vf[i-1]*J[i-1].T+J[i-1]*(vLag[0]-F*vf[i-1])*J[i-2].T)
        
        J0 = x0var*F.T*vp[0].I
        vLag[0] = vf[0]*J0.T+J[0]*(vLag[0]-F*vf[0])*J0.T
        xs0 = x0mean+J0*(xs[:,0]-xp[:,0])
        vs0 = x0var+J0*(vs[0]-vp[0])*J0.T

        S11 = xs[:,0]*xs[:,0].T + vs[0]
        S10 = xs[:,0]*xs0.T + vLag[0]
        S00 = xs0*xs0.T + x0var
        Syy = Yobs[:,0]*Yobs[:,0].T
        Syx = Yobs[:,0]*xs[:,0].T
        for i in range(N)[1:]:
            S11 += xs[:,i-1]*xs[:,i-1].T + vs[i-1]
            S10 += xs[:,i-1]*xs[:,i-2].T + vLag[i-1]
            S00 += xs[:,i-2]*xs[:,i-2].T + vs[i-2]
            Syy += Yobs[:,i-1]*Yobs[:,i-1].T
            Syx += Yobs[:,i-1]*xs[:,i-1].T
        
        return {
            "xp": DataFrame(xp.T),
            "vp": vp,
            "xf": DataFrame(xf.T),
            "vf": vf,
            "xs0": DataFrame(xs0.T),
            "xs": DataFrame(xs.T),
            "vs0": vs0,
            "vs": vs,
            "vLag": vLag,
            "S11": S11,
            "S10": S10,
            "S00": S00,
            "Syy": Syy,
            "Syx": Syx
            }

    def __calc_loglikelihood(self, values, observation):
        N = observation.shape[0] # number of time points

        return (-1/2) * (np.log(np.linalg.det(values["x0var"])) + \
                         np.trace(np.linalg.inv(values["x0var"])*(values["vs0"]+(values["xs0"]-values["x0mean"])*(values["xs0"]-values["x0mean"]).T)) + \
                         N*np.log(np.linalg.det(self.R)) + \
                         np.trace(np.linalg.inv(self.R)*(values["Syy"]+self.H*values["S11"]*self.H.T-values["Syx"]*self.H.T-self.H*values["Syx"].T)) + \
                         N*np.log(np.linalg.det(self.Q)) + \
                         np.trace(np.linalg.inv(self.Q)*(values["S11"]+self.F*values["S00"]*self.F.T-values["S10"]*self.F.T-self.F*values["S10"].T)) + \
                         (self.k+N*(self.k+self.p))*np.log(2*np.pi))

    def execute(self, observation):
        result = self.__pfs(observation)
        logllh = self.__calc_loglikelihood(result, observation)
        return {
            "xp": result["xp"],
            "vp": result["vp"],
            "xf": result["xf"],
            "vf": result["vf"],
            "xs0": result["xs0"],
            "xs": result["xs"],
            "vs0": result["vs0"],
            "vs": result["vs"],
            "logllh": logllh
            }
        


class EM__:
    def __init__(self, data, *sys_dim):
        # kalman instance for em
        self.data = data
        self.p = data.shape[1]
        self.k = sys_dim[0] if sys_dim else self.p
        self.N = data.shape[0]
        
        self.kl = Kalman(self.p, self.k)
        self.kl.set_data(self.data)
        # variable for em
        self.x0mean = self.kl.x0mean
        self.x0var = self.kl.x0var
        self.F = self.kl.F
        self.Q = self.kl.Q
        self.H = self.kl.H
        self.R = self.kl.R
        self.llh = []
        
    def __Estep(self):
        """ Private method: Expectation step of EM algorithm for SSM """

        # execute kalman's algorithm(prediction, filtering and smoothing)
        #Yobs = np.matrix(self.data.T)
        self.kl.F = self.F
        self.kl.H = self.H
        self.kl.Q = self.Q
        self.kl.R = self.R
        self.kl.x0mean = self.x0mean
        self.kl.x0var = self.x0var

        self.kl.pfs()
        self.kl.set_pfs_results()
        
        p = self.p
        k = self.k
        N = self.N
        F = self.F
        H = self.H
        Q = self.Q
        R = self.R
        xs = np.matrix(self.kl.xs.T)
        xs0 = np.matrix(self.kl.xs0.T)
        x0mean = self.kl.x0mean
        x0var = self.kl.x0var
        vs = self.kl.vs
        vs0 = self.kl.vs0
        S11 = self.kl.S11
        S10 = self.kl.S10
        S00 = self.kl.S00
        Syy = self.kl.Syy
        Syx = self.kl.Syx

        #print R.I
        #tmp = np.matrix(np.zeros([p,p]))
        #for i in range(N):
            #tmp += (Yobs[:,i]-H*xs[:,i])*(Yobs[:,i]-H*xs[:,i]).T + H*vs[i]*H.T

        """
        likelihood = (-1/2)*(np.log(np.linalg.det(x0var)) + \
                     np.trace(x0var.I*(vs0+(xs0-x0mean)*(xs0-x0mean).T)) + \
                     N*np.log(np.linalg.det(Q)) + \
                     np.trace(Q.I*(S11-S10*F.T-F*S10.T+F*S00*F.T)) + \
                     N*np.log(np.linalg.det(R)) + \
                     np.trace(R.I*tmp) + \
                     (k+N*(k+p))*np.log(2*np.pi))
        """

        logllh = np.log(np.linalg.det(x0var)) + \
                 np.trace(np.linalg.inv(x0var)*(vs0+(xs0-x0mean)*(xs0-x0mean).T)) + \
                 N*np.log(np.linalg.det(R)) + \
                 np.trace(np.linalg.inv(R)*(Syy+H*S11*H.T-Syx*H.T-H*Syx.T)) + \
                 N*np.log(np.linalg.det(Q)) + \
                 np.trace(np.linalg.inv(Q)*(S11+F*S00*F.T-S10*F.T-F*S10.T)) + \
                 (k+N*(k+p))*np.log(2*np.pi)

        print "\t(1)\t", np.trace(np.linalg.inv(x0var)*(vs0+(xs0-x0mean)*(xs0-x0mean).T))
        print "\t(2)\t", N*np.log(np.linalg.det(R))
        print "\t(3)\t", np.trace(np.linalg.inv(R)*(Syy+H*S11*H.T-Syx*H.T-H*Syx.T))
        print "\t(4)\t", N*np.log(np.linalg.det(Q))
        print "\t(5)\t", np.trace(np.linalg.inv(Q)*(S11+F*S00*F.T-S10*F.T-F*S10.T))

        logllh = (-1/2)*logllh
        #print logllh
        self.llh.append(logllh)


    def __Mstep(self):
        """ Private method: Maximization step of EM algorithm for SSM """
        Yobs = np.matrix(self.data.T)
        p = self.p
        k = self.k
        N = self.N
        #F = self.F
        H = self.H
        Q = self.Q
        S11 = self.kl.S11
        S10 = self.kl.S10
        S00 = self.kl.S00
        Syy = self.kl.Syy
        Syx = self.kl.Syx
        xs = np.matrix(self.kl.xs.T)
        xs0 = self.kl.xs0
        vs = self.kl.vs
        vs0 = self.kl.vs0
        self.F = S10*S00.I
        self.H = Syx*S11.I
        #self.Q = (S11 - S10*S00.I*S10.T)/N
        self.R = np.diag(np.diag(Syy - Syx*np.linalg.inv(S11)*Syx.T))/N
        self.x0mean = np.asarray(xs0.T)
        self.x0var = vs0

 
    def execute(self):
        """ Execute EM algorithm """
        count = 0
        diff = 100
        while diff>1e-3 and count<5000:
            print count,
            self.__Estep()
            self.__Mstep()
            if count>0: diff = abs(self.llh[count] - self.llh[count-1])
            print "\tllh:", self.llh[count]
            count += 1
        #print "#"
        #return 1



if __name__ == "__main__":
    print "kalman.py: Called in main process."

    if 0:
        obs_p = 2
        sys_k = 2
        N = 20
        kl = Kalman(obs_p,sys_k)
        data = kl.ssm.gen_data(N)
        #kl.set_data(data[1])
        #kl.pfs()
        #for i in range(100): kl.pfs()
        em = EM(data[1], sys_k)
        em.execute()

        if 1:
            fig, axes = plt.subplots(np.int(np.ceil(sys_k/3.0)), 3, sharex=True)
            j = 0
            for i in range(3):
                while j<sys_k:
                    if sys_k<=3:
                        axes[j%3].plot(data[0][j], "k--", label="obs")
                        axes[j%3].plot(em.kl.xp[j], label="prd")
                        axes[j%3].legend(loc="best")
                        axes[j%3].set_title(j)
                    else:
                        axes[i, j%3].plot(data[0][j], "k--", label="obs")
                        axes[i, j%3].plot(em.kl.xp[j], label="prd")
                        axes[i, j%3].legend(loc="best")
                        axes[i, j%3].set_title(j)
                    j += 1
                    if j%3 == 2: break

            fig.show()
        
        #loss = data[0]-em.kl.xs
        #plt.plot(loss)
        #plt.plot(em.llh)
        #plt.show()
