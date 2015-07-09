import numpy as np
import pandas as pd
#from pandas import Series, DataFrame

def sp_matrix(i, j):
	mat = np.random.randn(i,j)
	mat[abs(mat)<1] = 0
	#mat = mat.map(lambda x: 0 if np.abs(x)>0.5 else x)
	return np.matrix(mat)
