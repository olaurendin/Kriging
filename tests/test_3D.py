import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import norm, probplot
from mpl_toolkits.mplot3d import Axes3D

L, n = 2, 400
x = np.linspace(-L, L, n)
y = x.copy()
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

fig = plt.figure()
ax = Axes3D(fig)

print(x,y)
print(x.shape, y.shape,X.shape, Y.shape, Z.shape)


# ax.plot_wireframe(X, Y, Z, rstride=40, cstride=40)
ax.plot_surface(X, Y, Z, rstride=40, cstride=40, color='m')
plt.show()
