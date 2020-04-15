import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
f = open(sys.argv[1],'r')
#print(sys.argv[1])
scale = 100
x_data = []
y_data = []
z_data = []
for lines in f:
    x, y, z = lines.split()
    x_data.append(float(x)*scale)
    y_data.append(float(y)*scale)
    z_data.append(float(z)*scale)
ax.scatter(x_data, y_data, z_data, s=11)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(False)
plt.show()
