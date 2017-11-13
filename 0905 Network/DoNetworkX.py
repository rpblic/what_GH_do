import os
os.chdir("C:\\Studying\\GrowthHackers\\0905 Network")
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Data_Structure import data_making, top_n

grp= nx.Graph()
grp.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])
grp.add_node('g', author= 'team_5')

grp.node['a']['author']= 'Euler'

grp.add_edge('a', 'b', weight= 0.5, color= 'red')
edges_to_add= [('b', 'c'), ('c', 'd'), ('b', 'd'), ('b', 'g'), ('e', 'b'), ('e', 'f'), ('e', 'g'), ('g', 'c'), ('g', 'd')]
grp.add_edges_from(edges_to_add)

print(grp.nodes(), type(grp.nodes()))
print(grp.nodes(data=True))
print(grp.edges(data=True))

"""As we learned"""
grpmatx= nx.to_numpy_matrix(grp)
print('Matrix: ', grpmatx)      #graph를 나타내는 numpy matrix 도출
print('Degree: ', nx.degree(grp))     #graph의 차수 도출
deg_sum= float(sum(grp.degree().values()))
print('Average Node Degree: ', deg_sum/nx.number_of_nodes(grp))
print('Density: ', nx.density(grp))

print('Betweenness: ', nx.betweenness_centrality(grp))
print('Closeness: ', nx.closeness_centrality(grp))
print('Degree: ', nx.degree_centrality(grp))

[eigvalue, eigvec]= np.linalg.eig(grpmatx)
print(eigvalue)
print(eigvec)
print('Eigenvalue_based: ', nx.eigenvector_centrality(grp))

"""Another Statistic values"""

"""Distance between Nodes"""

print("Communicability: ", nx.communicability(grp)['a'])
'''Our strategy here is to make longer walks have lower contributions to the
communicability function than shorter ones. While a shortest path represents only a
single path that communicates both nodes, our approach considers all ways in which we
can reach the target node q starting our walk at the node p. As some of these "detour"
can be very long, the summation is weighted in decreasing order of the length of the walk.'''

print('Average Path Length: ', nx.average_shortest_path_length(grp))

"""Cluster between Nodes"""

print('Clustering for one Node: ', nx.clustering(grp))
print('Avergage Clustering Coefficient: ', nx.average_clustering(grp))
'''Algorithms to characterize the number of triangles in a graph.'''

print('Transitivity: ', nx.transitivity(grp))
'''(possible Triangles)/(Triads)'''

print('Connectivity value: ', nx.node_connectivity(grp))
print('Connectivity for all Nodes: ', nx.all_pairs_node_connectivity(grp))
'''Pairwise or local node connectivity between two distinct and nonadjacent nodes is
the minimum number of nodes that must be removed (minimum separating cutset) to disconnect
them. By Menger’s theorem, this is equal to the number of node independent paths (paths that
share no nodes other than source and target). Which is what we compute in this function.'''

print('Core number: ', nx.core_number(grp))
'''Return the core number for each vertex.
A k-core is a maximal subgraph that contains nodes of degree k or more.
The core number of a node is the largest value k of a k-core containing that node.'''

print('Cycle basis: ', nx.cycle_basis(grp))
'''A basis for cycles(loops) of a network is a minimal collection of cycles such that
any cycle in the network can be written as a sum of cycles in the basis.'''

"""Dispersion, and others"""

print('Dispersion: ', nx.dispersion(grp))
'''‘Low dispersion’ – the quality that was associated with couples – indicates not only that
two people have a large number of mutual friends, but also that these mutual friends knew
one another.'''

print('Assortativity: ', nx.degree_assortativity_coefficient(grp))
'''Assortativity, or assortative mixing is a preference for a network's nodes to attach
to others that are similar in some way. Though the specific measure of similarity may vary,
network theorists often examine assortativity in terms of a node's degree.'''

print('Pagerank: ', nx.pagerank(grp, alpha= 0.85, max_iter= 100, tol= 1e-06))
'''PageRank computes a ranking of the nodes in the graph G based on the structure
of the incoming links. It was originally designed as an algorithm to rank web pages.'''

# print('Hits: ', nx.hits(grp))
'''DIRECTED NETWORK ONLY. HITS algorithm computes two numbers for a node. Authorities
estimates the node value based on the incoming links. Hubs estimates the node value
based on outgoing links.'''

pos= nx.shell_layout(grp)
print(pos)
'''take a vector of each nodes'''

nx.draw(grp, pos, node_size= [(i*500)**2 for i in nx.pagerank(grp).values()])
nx.draw_networkx_labels(grp, pos)
plt.show()

nx.write_gml(grp, "GH_Network.gml")
