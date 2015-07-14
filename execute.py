import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from var import *

def draw_heatmap(data, **labels):
	fig, ax = plt.subplots(figsize=(10, 10))
	heatmap = ax.pcolor(data, cmap=plt.cm.Reds)
	ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
	ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
	ax.invert_yaxis()
	ax.xaxis.tick_top()

	if labels:
		ax.set_xticklabels(labels["row"], minor=False)
		ax.set_yticklabels(labels["column"], minor=False)

	plt.show()
	#plt.savefig('image.png')
	
	return heatmap

def draw_heatmap2(x, y):
	heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	plt.figure()
	plt.imshow(heatmap, extent=extent)
	plt.show()
	#plt.savefig('image.png')
	

if __name__ == "__main__":
	filename = "./ignr/data/exchange.dat"
	#data = np.loadtxt(filename, delimiter="\t")
	df = pd.read_table(filename, index_col="datetime")
	df = df.ix[df.index.map(lambda x: x.endswith("00:00")), :]
	df_price = df.ix[:, df.columns.map(lambda x: x.endswith("PRICE"))]
	df_value = df.ix[:, df.columns.map(lambda x: x.endswith("VALUE"))]

	# SVAR
	if 1:
		#svar = SparseVAR()
		#svar.set_data(df_price)
		##svar.SVAR(5)
		#B = svar.GCV()
		draw_heatmap(np.array(B))
