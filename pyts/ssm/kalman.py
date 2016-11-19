#coding: utf-8
"""
Kalman's algorithm(prediction, filtering, smoothing)

Author: mammaru <mauma1989@gmail.com>

"""
import numpy as np
#import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from .base import StateSpaceModel
from ..base import EMmixin


class SSMKalman(StateSpaceModel, EMmixin):
    def __predict(self, values):
        return {
            "xp": self.F*values["xf"],
            "vp": self.F*values["vf"]*self.F.T + self.Q
            }

    def __filter(self, values):
        #print values["vp"]
        K = values["vp"]*self.H.T*(self.H*values["vp"]*self.H.T + self.R).I
        return {
            "xf": values["xp"] + K*(values["obs"] - self.H*values["xp"]),
            "vf": values["vp"] - K*self.H*values["vp"],
            "K": K
            }

    def __smooth(self, values):
        J = values["vf-1"]*self.F.T*values["vp"].I
        return {
            "xs": values["xf-1"] + J*(values["xs"] - values["xp"]),
            "vs": values["vf-2"] + J*(values["vs"] - values["vp"])*J.T,
            "J": J
            }

    def __prediction_filtering(self, data):
        print "prediction_filtering"
        vp = []
        vf = []
        N = data.shape[0] # number of time points
        obs = np.matrix(data.T)
        xp = np.matrix(self.F*self.x0mean)
        vp.append(self.F*self.x0var*self.F.T + self.Q)
        # filtering
        values = self.__filter({"xp": xp, "vp": vp[0], "obs": obs[:,0]})
        K = values["K"]
        xf = xp[:,0]+K*(obs[:,0] - self.H*xp[:,0])
        xf = np.hstack((xf, values["xf"]))
        vf.append(values["vf"])
        # prediction
        values = self.__predict(values)
        xp = np.hstack((xp, values["xp"]))
        vp.append(values["vp"])
        for i in range(N)[1:]:
            print i
            # filtering
            values = self.__filter({"xp": xp, "vp": vp[i], "obs": obs[:,i]})
            K = values["K"]
            xf = np.hstack((xf, values["xf"]))
            vf.append(values["vf"])
            # prediction
            values = self.__predict(values)
            xp = np.hstack((xp, values["xp"]))
            vp.append(values["vp"])

        return {
            "xp": xp,
            "vp": vp,
            "xf": xf,
            "vf": vf,
            "K": K
            }

    def __smoothing(self, data, pf_values):
        vs = []
        N = data.shape[0] # number of time points
        #obs = np.matrix(data.T)
        J = [np.matrix(np.zeros([k,k]))]
        xs = pf_values["xf"][:,N-1]
        vs.insert(0, pf_values["vf"][N-1])
        vLag.insert(0, self.F*pf_values["vf"][N-2] - pf_values["K"]*self.H*pf_values["vf"][N-2])
        for i in reversed(range(N)[1:]):
            print i
            values = self.__smooth({"xp": pf_values["xp"][:,i],
                                    "xf-1": pf_values["xf"][:,i-1],
                                    "xs": xs[:,0],
                                    "vp": pf_values["vp"][i],
                                    "vf-1": pf_values["vf"][i-1],
                                    "vf-2": pf_values["vf"][i-2],
                                    "vs": vs[0]})
            J.insert(0, values["J"])
            xs = np.hstack((values["xs"], xs))
            vs.insert(0, values["vs"])       
        for i in reversed(range(N)[2:]):
            print i
            vLag.insert(0, pf_values["vf"][i-1]*J[i-1].T + J[i-1]*(vLag[0] - F*pf_values["vf"][i-1])*J[i-2].T)
        
        J0 = self.x0var*self.F.T*pf_values["vp"][0].I
        vLag[0] = pf_values["vf"][0]*J0.T + J[0]*(vLag[0] - F*pf_values["vf"][0])*J0.T
        xs0 = self.x0mean + J0*(xs[:,0] - pf_values["xp"][:,0])
        vs0 = self.x0var + J0*(vs[0] - pf_values["vp"][0])*J0.T

        return {
            "xs0": xs0,
            "xs": xs,
            "vs0": vs0,
            "vs": vs,
            "vLag": vLag
            }

    def __pfs(self, data):
        """ body of the kalman's method - prediction, filtering and smoothing """
        print "pfs"
        vs = []
        N = data.shape[0] # number of time points
        print N
        obs = np.matrix(data.T)
        pf_result = self.__prediction_filtering(data)
        s_result = self.__smoothing(data, pf_result)

        S11 = s_result["xs"][:,0]*s_result["xs"][:,0].T + s_result["vs"][0]
        S10 = s_result["xs"][:,0]*s_result["xs0"].T + s_result["vLag"][0]
        S00 = s_result["xs0"]*s_result["xs0"].T + self.x0var
        Syy = obs[:,0]*obs[:,0].T
        Syx = obs[:,0]*s_result["xs"][:,0].T
        for i in range(N)[1:]:
            print i
            S11 += s_result["xs"][:,i-1]*s_result["xs"][:,i-1].T + s_result["vs"][i-1]
            S10 += s_result["xs"][:,i-1]*s_result["xs"][:,i-2].T + s_result["vLag"][i-1]
            S00 += s_result["xs"][:,i-2]*s_result["xs"][:,i-2].T + s_result["vs"][i-2]
            Syy += obs[:,i-1]*obs[:,i-1].T
            Syx += obs[:,i-1]*s_result["xs"][:,i-1].T
        
        return {
            "xp": DataFrame(pf_result["xp"].T),
            "vp": pf_result["vp"],
            "xf": DataFrame(pf_result["xf"].T),
            "vf": pf_values["vf"],
            "xs0": DataFrame(s_result["xs0"].T),
            "xs": DataFrame(s_result["xs"].T),
            "vs0": s_result["vs0"],
            "vs": s_result["vs"],
            "vLag": s_result["vLag"],
            "S11": S11,
            "S10": S10,
            "S00": S00,
            "Syy": Syy,
            "Syx": Syx
            }

    def __calc_loglikelihood(self, values, observation):
        N = observation.shape[0] # number of time points

        return (-1/2) * (np.log(np.linalg.det(values["x0var"])) + \
                         np.trace(np.linalg.inv(values["x0var"])*(values["vs0"] + (values["xs0"] - values["x0mean"])*(values["xs0"] - values["x0mean"]).T)) + \
                         N*np.log(np.linalg.det(self.R)) + \
                         np.trace(np.linalg.inv(self.R)*(values["Syy"] + self.H*values["S11"]*self.H.T - values["Syx"]*self.H.T - self.H*values["Syx"].T)) + \
                         N*np.log(np.linalg.det(self.Q)) + \
                         np.trace(np.linalg.inv(self.Q)*(values["S11"] + self.F*values["S00"]*self.F.T - values["S10"]*self.F.T - self.F*values["S10"].T)) + \
                         (self.k + N*(self.k + self.p))*np.log(2*np.pi))

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

    def expectation(self, observation):
        print "expectation"
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
            "S11": result["S11"],
            "S10": result["S10"],
            "S00": result["S00"],
            "Syy": result["Syy"],
            "Syx": result["Syx"],
            "logllh": logllh
            }

    def maximization(self, values, observation):
        N = observation.shape[0] # number of time points
        self.F = values["S10"]*values["S00"].I
        self.H = values["Syx"]*values["S11"].I
        #self.Q = (values["S11"] - values["S10"]*values["S00"].I*values["S10"].T)/N
        self.R = np.diag(np.diag(values["Syy"] - values["Syx"]*np.linalg.inv(values["S11"])*values["Syx"].T))/N
        self.x0mean = np.asarray(values["xs0"].T)
        self.x0var = values["vs0"]

    def em_step_delegate(self, data):
        values = self.expectation(data)
        self.maximization(values, data)
        return values["llh"]
        

