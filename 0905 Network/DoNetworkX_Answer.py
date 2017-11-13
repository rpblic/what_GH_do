import os
os.chdir("C:\\Studying\\GrowthHackers\\0905 Network")
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Data_Structure import data_making, top_n

grp= nx.Graph()

node, q1_edge, q2_edge= data_making('.\\최종 전처리 데이터.csv')
grp.add_nodes_from(node)
grp.add_edges_from(q1_edge)

grp.remove_edges_from(grp.selfloop_edges())

# print(top_n(nx.eigenvector_centrality(grp), 3))
# print(top_n(nx.hits(grp)[0], 3))
# print(top_n(nx.hits(grp)[1], 3))
print(top_n(nx.communicability(grp)['18'], 3))
