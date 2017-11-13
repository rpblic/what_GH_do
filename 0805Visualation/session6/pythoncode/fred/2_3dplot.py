import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
mu = 2
sigma = 3
a = np.random.normal(mu, sigma, 100000)
b = np.random.normal(mu, sigma, 100000)


heatmap, xedges, yedges = np.histogram2d(a, b, bins=50)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_data, y_data = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = heatmap.flatten()






ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data ,alpha=0.7)
plt.show()