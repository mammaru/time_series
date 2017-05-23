#coding: utf-8
import numpy as np
import pandas as pd
#from pandas import Series, DataFrame
from matplotlib import pyplot as plt
import networkx as nx

def sp_matrix(i, j):
    mat = np.random.randn(i,j)
    mat[abs(mat)<1] = 0
    #mat = mat.map(lambda x: 0 if np.abs(x)>0.5 else x)
    return np.matrix(mat)

def heatmap(data, **labels):
    d = np.array(data)
    fig, axis = plt.subplots(figsize=(10, 10))
    heatmap = axis.pcolor(d, cmap=plt.cm.Reds)
    axis.set_xticks(np.arange(d.shape[0])+0.5, minor=False)
    axis.set_yticks(np.arange(d.shape[1])+0.5, minor=False)
    axis.invert_yaxis()
    axis.xaxis.tick_top()
    if labels:
        print "bbbbbbbbbbb!"
        axis.set_xticklabels(labels["row"], minor=False)
        axis.set_yticklabels(labels["column"], minor=False)
    elif isinstance(data, pd.DataFrame):
        print "aaaaaaaaa!"
        axis.set_xticklabels(data.index, minor=False)
        axis.set_yticklabels(data.columns, minor=False)        
    fig.show()
    return heatmap

def digraph(data):
    DG = nx.DiGraph()
    idxs = data.index.tolist()
    #print idxs
    for idx_from in idxs:
        for idx_to in idxs:
            #print idx_from, idx_to
            #print data.loc[idx_from, idx_to]
            if data.ix[idx_from,idx_to]!=0:
                #print B.ix[idx_from,idx_to]
                DG.add_edge(idx_from, idx_to, weight=data.ix[idx_from,idx_to])
    pos = nx.spring_layout(DG)
    nx.draw_networkx_nodes(DG, pos, node_size = 100, node_color = 'w')
    nx.draw_networkx_edges(DG, pos)
    nx.draw_networkx_labels(DG, pos, font_size = 12, font_family = 'sans-serif', font_color = 'r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def test(data):
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

