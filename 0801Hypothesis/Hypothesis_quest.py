import numpy as np
import scipy.stats
import csv
import os
# print(os.)
os.chdir("C:\\Studying\\GrowthHackers\\0801Hypothesis")

class Hypothesis(object):
    def __init__(self, filename):
        self.filename= filename

    def takedata(self):
        with open(self.filename, "rt", newline='', encoding='UTF-8') as f:
            data= csv.reader(f, delimiter=',')
            arr=[]
            for row in data:
                arr.append(row)
            arr= np.array(arr)
            return arr
            # print(data2)

    def take2data(self, int1, int2, title=None, dtype=None):
        data2= self.takedata()
        data2= data2[:, [int1, int2]]
        if title!=None:
            data2= data2[1:,:]
        try:
            data2= data2.astype(np.float)
        except:
            pass
        return data2

    def fligner(self, data2):
        if data2.shape[1]==2:
            return scipy.stats.fligner(data2[:,0], data2[:,1])
        else:
            raise ValueError('take2data first.')

    def ttest_ind(self, data2):     #독립표본 t-검정
        if self.fligner(data2)[1]>=0.05:       #등분산
            return scipy.stats.ttest_ind(data2[:,0],data2[:,1], equal_var=True)
        else:       #이분산
            return scipy.stats.ttest_ind(data2[:,0],data2[:,1], equal_var=False)

    def normtest(self, data2, sigma=1, mu=0):
        self.fligner(data2)
        subdata= data2[:,0] - data2[:,1]
        z= (subdata.mean() - mu)/np.sqrt(sigma/len(subdata))
        print("Z value, p value")
        return z, 2*scipy.stats.norm().sf(np.abs(z))

hypt= Hypothesis("mtcars.csv").take2data(5,7, title='Y', dtype='Y')
print(Hypothesis("mtcars.csv").ttest_ind(hypt))
print(Hypothesis("mtcars.csv").normtest(hypt))
#flexible type means that there exists multiple types in one matrix
# matx= [[1,2,3,4], [5,6,7,8]]
# arr= np.array(matx)
# print(arr.mean())
# arr2= arr[:,[0,2]]
# print(arr2)
