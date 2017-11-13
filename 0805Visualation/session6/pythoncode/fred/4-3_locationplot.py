import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import numpy as np

f=open('./gpsdata.txt','r')

lines=f.readlines()
xlist = []
ylist = []
for line in lines:
	data = line.strip().split('\t')
	x = float(data[1])/pow(10,7)
	y = float(data[2])/pow(10,7)
	if x > 120 and x < 135 and y < 38 and y > 33:
		xlist.append(x)
		ylist.append(y)
f.close()
plt.scatter(xlist, ylist, edgecolor='none', c='g', alpha=0.25)

f2=open('./ch.txt', 'r')
lines=f2.readlines()
xlist = []
ylist = []
for line in lines:
	data = line.strip().split('\t')
	x = float(data[1])/pow(10,7)
	y = float(data[2])/pow(10,7)
	if x > 120 and x < 135 and y < 38 and y > 33:
		xlist.append(x)
		ylist.append(y)


plt.scatter(xlist, ylist, edgecolor='none', c='r', alpha=0.25)
#ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data ,alpha=0.7)
plt.show()