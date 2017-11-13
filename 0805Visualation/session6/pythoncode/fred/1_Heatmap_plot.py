import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mu = 0
sigma = 3
a = np.random.normal(mu, sigma, 300000)
b = np.random.normal(mu, sigma, 300000)

#plt.scatter(a, b, edgecolor='none', c='g', alpha=0.25)

heatmap, xedges, yedges = np.histogram2d(a, b, bins=200)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


cmap = plt.cm.rainbow

cmaplist = [cmap(i) for i in range(cmap.N)]

cmaplist[0] = (1.0,1.0,1.0,1.0)

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

bounds = np.linspace(0,100,101)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cax = plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
cbar = plt.colorbar(cax, ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
cbar.ax.set_yticklabels(['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '>100']) 




plt.show()



