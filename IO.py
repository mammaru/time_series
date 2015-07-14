import numpy as np
import pandas as pd

filename = "./ignr/data/exchange.dat"
#data = np.loadtxt(filename, delimiter="\t")
data = pd.read_table(filename, index_col="datetime")

