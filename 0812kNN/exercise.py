import numpy as np
from collections import Counter
import os
os.chdir('C:\\Studying\\GrowthHackers\\0812kNN')

def dist(u, v):
    return sum((np.array(u)-np.array(v))**2)**0.5

# print(dist([1,3], [2,4]))

with open('cities.txt', 'rt') as opener:
    lines= [line[:-1] for line in opener]
cities=[]
for i in range(len(lines)):
    lines[i]= lines[i].split(',')


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

def knn_classify(point, label, new):
    dist_sort= sorted(label, key= lambda x: dist(x[0], new))
    kth_near= [lang for _, lang in dist_sort[:k]]
    return majority(kth_near)
