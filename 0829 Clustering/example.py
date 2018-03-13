import os
os.chdir("C:\\Studying\\GrowthHackers\\0829 Clustering")

from linear_algebra import squared_distance, vector_mean, distance
import math, random
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k):
        self.k= k       #number of clusters
        self.means= None        #cluster의 중심점
        self.assignments= None      #각 데이터 포인트들의 군집화 결과

    def classify(self, inp):        #inp: 벡터 데이터 포인트
        return min(list(range(self.k)), key= lambda i: squared_distance(inp, self.means[i]))

    def train(self, inps):      #inps: 벡터 데이터 포인트들
        self.means= random.sample(inps, self.k)
        self.assignments= None

        while True:
            new_assignments= list(map(self.classify, inps))

            if self.assignments== new_assignments:
                return

            self.assignments= new_assignments

            for i in range(self.k):
                i_points= [p for p, a in zip(inps, self.assignments) if a==i]
                #Avoid divide-by-zero if i_points is empty
                if i_points:
                    self.means[i]= vector_mean(i_points)

inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
random.seed(0) # so you get the same results as me
clusterer = KMeans(3)
clusterer.train(inputs)

print(clusterer.means)
print(clusterer.assignments)

for i in range(len(inputs)):
    xs, ys = zip(inputs[i])
    if clusterer.assignments[i] == 0:
        plt.scatter(xs, ys, marker = 'D', color = 'r')
    elif clusterer.assignments[i] == 1:
        plt.scatter(xs, ys, marker = 'o', color = 'g')
    else:
        plt.scatter(xs, ys, marker = '*', color = 'b')

plt.show()
