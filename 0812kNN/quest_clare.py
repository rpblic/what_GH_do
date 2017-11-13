import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Studying\\GrowthHackers\\0812kNN')

with open('Iris.csv','rt') as opener:
    lines= [line[:-1] for line in opener]
    lines= lines[1:]

# print(lines[:5])

def dist(u, v):
    return sum((np.array(u)-np.array(v))**2)**0.5
def majority(language):
    vote_count= Counter(language)
    winner, winner_count= vote_count.most_common(1)[0]
    num_winners= len([count for count \
    in vote_count.values() \
    if count== winner_count])
    if num_winners==1:
        return winner
    else:
        return majority(language[:-1])
def knn_classify(point, label, new, k=10):
    dist_sort= sorted(label, key= lambda x: dist(x[0], new))
    kth_near= [lang for _, lang in dist_sort[:k]]
    return majority(kth_near)
def plot_iris_sepal1():
    plots = { "Iris-setosa" : ([], []), "Iris-versicolor" : ([], []), "Iris-virginica" : ([], []) }
    markers = { "Iris-setosa" : "o", "Iris-versicolor" : "s", "Iris-virginica" : "^" }
    colors  = { "Iris-setosa" : "r", "Iris-versicolor" : "b", "Iris-virginica" : "g" }
    for (sepal_length, sepal_width), species in iris_sepal1:
        plots[species][0].append(sepal_length)
        plots[species][1].append(sepal_width)
    for species, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[species], marker=markers[species], label=species)
    plt.legend(loc=0)
    plt.axis([4,8.5,1.5,5])                 #x축 y축 범위 설정
    plt.title("Iris Species")
    plt.xlabel("Sepal_length")
    plt.ylabel("Sepal_width")
    plt.show()
def classify_and_plot_grid(k=1):
    plots = { "Iris-setosa" : ([], []), "Iris-versicolor" : ([], []), "Iris-virginica" : ([], []) }
    markers = { "Iris-setosa" : "o", "Iris-versicolor" : "s", "Iris-virginica" : "^" }
    colors  = { "Iris-setosa" : "r", "Iris-versicolor" : "b", "Iris-virginica" : "g" }
    for sepal_length in range(40, 85):
        for sepal_width in range(15, 50):
            predicted_species = knn_classify(k, iris_sepal2, [sepal_length, sepal_width])
            plots[predicted_species][0].append(sepal_length)
            plots[predicted_species][1].append(sepal_width)
    for species, (x, y) in plots.items():
        plt.scatter(x, y, color=colors[species], marker=markers[species],
                          label=species)
    plt.legend(loc=0)
    plt.axis([40,85,15,50])
    plt.title(str(k) + "-Nearest Neighbor Iris Species")
    plt.xlabel("Sepal_length x 10")
    plt.ylabel("Sepal_width x 10")
    plt.show()

result=[]
for line in lines:
    listline= line.split(',')
    listline= listline[1:]
    listline= tuple([float(listline[0]), \
    float(listline[1]), float(listline[2]), float(listline[3]), listline[4]])
    result.append(listline)

# print(result[:5])

iris_sepal1= [([a,b], e) for a, b, c, d, e in result]
iris_sepal2 = [([a*10,b*10],e) for a,b,c,d,e in result]

# plot_iris_sepal1()
classify_and_plot_grid()
