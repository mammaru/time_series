import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from svar import *
from kalman import *

def draw_heatmap(data, **labels):
	fig, axis = plt.subplots(figsize=(10, 10))
	heatmap = axis.pcolor(data, cmap=plt.cm.Reds)
	axis.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
	axis.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
	axis.invert_yaxis()
	axis.xaxis.tick_top()

	if labels:
		axis.set_xticklabels(labels["row"], minor=False)
		axis.set_yticklabels(labels["column"], minor=False)

	fig.show()
	#plt.savefig('image.png')
	
	return heatmap



if __name__ == "__main__":
	if 0:
		filename = "./ignr/data/exchange.dat"
		#data = np.loadtxt(filename, delimiter="\t")
		df = pd.read_table(filename, index_col="datetime")
		df.index = pd.to_datetime(df.index) # convert index into datetime
		#hourly = df.resample("H", how="mean") # hourly
		daily = df.resample("D", how="mean") # daily
		price = daily.ix[:, daily.columns.map(lambda x: x.endswith("PRICE"))]
		volume = daily.ix[:, daily.columns.map(lambda x: x.endswith("VOLUME"))]

	# SVAR
	if 0:
		data = price
		svar = SparseVAR()
		svar.set_data(data)
		#svar.regression(5)
		
		B = svar.GCV()
		B = DataFrame(B.T, index=data.columns, columns=data.columns)
	if 0:
		draw_heatmap(np.array(B))
	if 1:
		DG = nx.DiGraph()
		idxs = data.columns
		for idx_from in idxs:
			for idx_to in idxs:
				if abs(B[idx_from][idx_to])>0: DG.add_edge(idx_from, idx_to, weight=B[idx_from][idx_to])
		pos = nx.spring_layout(DG)
		nx.draw_networkx_nodes(DG, pos, node_size = 100, node_color = 'w')
		nx.draw_networkx_edges(DG, pos)
		nx.draw_networkx_labels(DG, pos, font_size = 12, font_family = 'sans-serif', font_color = 'r')
		plt.xticks([])
		plt.yticks([])
		plt.show()

		

	# kalman and EM
	if 0:
		sys_k = 5
		em = EM(price, sys_k)
		em.execute()

   	if 0:
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
