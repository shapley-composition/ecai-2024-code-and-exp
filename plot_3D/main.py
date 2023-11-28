import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import ConvexHull

# create the figure
fig = plt.figure()

# add axes
ax = fig.add_subplot(111,projection='3d')

# yy, zz = np.meshgrid(np.linspace(-5,5,10), np.linspace(-5,5,10))
# xx = np.linspace(0,5,10)

# # plot the plane
# ax.plot_surface(xx, yy, zz, alpha=0.5)

xx, zz = np.meshgrid(np.linspace(0,5,100), np.linspace(-5,5,100))
yy = -xx/sqrt(3)

print(yy.shape)

# plot the plane
ax.plot_surface([0, 0, 0], [0, 1,0], [0, 1 ,1], alpha=0.5)


# xx, yy = np.meshgrid(np.linspace(0,5,100), np.linspace(-5,5,100))
# zz = yy/sqrt(2)

# # plot the plane
# ax.plot_surface(xx, yy, zz, alpha=0.5)



plt.show()
