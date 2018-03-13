import networkx as nx
from networkx import *
import matplotlib.pyplot as plt
import numpy as np



f = open('6_subwaydata.txt','r')
lineslist = f.readlines()
linename = []
station = []
code = []
G=nx.Graph()


for line in lineslist:
	row = line.strip().split('\t')
	linename.append(row[1])
	station.append(row[0])
	code.append(row[2])

for i in station:
	G.add_node(i)
#print(G.nodes(data=True))

for i in list(range(len(code))):
	if i ==1:
		continue
	try:
		if int(code[i]) - int(code[i-1]) == 1:
			G.add_edge(station[i],station[i-1])
	except:
		try:
			if int(code[i][1:]) - int(code[i-1][1:]) == 1:
				G.add_edge(station[i],station[i-1])
		except:
			#print(code[i])
			pass


#1st line
G.add_edge(station[603],station[604])
G.add_edge(station[603],station[606])

G.add_edge(station[617],station[618])
G.add_edge(station[617],station[619])

G.add_edge(station[41],station[601])



#2nd line
G.add_edge(station[62],station[112])

G.add_edge(station[72],station[73])
G.add_edge(station[73],station[74])
G.add_edge(station[74],station[75])
G.add_edge(station[75],station[76])
G.add_edge(station[72],station[77])

G.add_edge(station[99],station[100])
G.add_edge(station[100],station[101])
G.add_edge(station[101],station[102])
G.add_edge(station[102],station[103])
G.add_edge(station[99],station[104])

#5th line
G.add_edge(station[243],station[641])



#6nd line
G.add_edge(station[249],station[254])
G.add_edge(station[249],station[255])
G.add_edge(station[393],station[392])
G.add_edge(station[394],station[393])
G.remove_edge(station[254],station[255])

G.remove_edge(station[113],station[114])
G.remove_edge(station[121],station[122])
G.remove_edge(station[337],station[336])

f2 = open("6_additional.txt",'r')
lines = f2.readlines()
for line in lines:
	row = line.strip().split('\t')
	G.add_edge(row[0],row[1])


degree = []
for i in G.nodes():
	degree.append(len(G.neighbors(i)))


nx.write_gexf(G, "6_subway.gexf")
