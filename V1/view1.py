# import show3d
import numpy as np
import sys
# a=np.loadtxt(sys.argv[1])
# show3d.showpoints(a)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a=np.loadtxt(sys.argv[1])
x,y,z = a.T
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x, y, z)
plt.show()
