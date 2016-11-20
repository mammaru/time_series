#coding: utf-8
#import numpy as np
#import pandas as pd
#from pandas import DataFrame

EM_THRESHOLD = 1e-1
EM_ITERATION_MAXIMUM_COUNT = 5000

class TimeSeriesModel(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''

    def initialize(**args):
        raise ValueError("Model initialization failed.")


class EMmixin:
    def setData(self, obs):
        self.data = obs

    def em_step_delegate(self, data):
        return self.em_step(data)

    def em_step(self, data):
        assert 0, "Not implemented em_step method in child class"
    
    def expectation(self):
        assert 0, "Not implemented expectation method in child class"

    def maximization(self):
        assert 0, "Not implemented maximization method in child class"


class EM:
    def __init__(self, threshold = EM_THRESHOLD, max_count = EM_ITERATION_MAXIMUM_COUNT):
        self.threshold = threshold
        self.iteration_max_count = max_count

    def __call__(self, _model, _data):
        self.execute(_model, _data)

    def execute(self, _model, _data):
        """ Execute EM algorithm """
        print "Start EM algorithm"
        self.model = _model
        #self.model.setData(_data)        
        count = 0
        diff = 100
        llh = []
        while diff>self.threshold and count<self.iteration_max_count:
            print count,
            llh.append(self.model.em_step_delegate(_data))
            if count>0: diff = abs(llh[count] - llh[count-1])
            print "likelihood:", llh[count]
            count += 1
        print ""
        return llh
