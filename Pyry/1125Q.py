import numpy as np
import os
os.chdir('C:\\Studying\\myvenv\\GrowthHackers\\Pyry')

arr2= np.arange(9)
arr2= arr2.reshape((3,3))

arr4= np.random.randint(0, 100, (5,5))
avg_arr4= np.sum(arr4)/np.size(arr4)
std_arr4= np.std(arr4)
min_arr4= np.min(arr4)
max_arr4= np.max(arr4)
arr4_normalize= (arr4- min_arr4)/(max_arr4- min_arr4)

arr6= np.genfromtxt('.\\[Module 2] Numpy & Pandas\\stock-data.csv', dtype=float, delimiter=',')
arr6_x, arr6_y= arr6[:, :4], arr6[:, 4:]
print(arr2, '\n', arr4,'\n', arr4_normalize, '\n', arr6_x, '\n', arr6_y, '\n', arr6_x.shape, arr6_y.shape)
