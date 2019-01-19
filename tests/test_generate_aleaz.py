from random import uniform
import numpy as np
from scipy.spatial.distance import cdist


xmin, xmax, ymin, ymax = [0,10,0,10]
avgalt = 100
nb_deposits=1
x = np.linspace(1.0,2.0,2)
y = np.linspace(1.0,2.0,2)
X,Y = np.meshgrid(x,y)
X2, Y2 = X.flatten(), Y.flatten()
# print(X,Y, X2, Y2)
points = np.vstack((X2, Y2, avgalt*np.ones(X2.shape))).T
print(points.shape, points)
# Generation of the fake deposits
deposits = np.array([[uniform(xmin,xmax), uniform(ymin, ymax), avgalt, 1] for i in range(nb_deposits)])
deposits = np.array([[1,1.5, avgalt, 1] for i in range(nb_deposits)])

# Calculation of the distances between each position of the robot and each deposit :
dists = cdist(points, deposits[:,:3])
# Addition of a gaussian noise in the background (optionnal)
mu, sigma, scale = 1, 0.25, 1 # mean and standard deviation of the
background = scale*np.random.normal(mu, sigma, len(dists))
# Calculation of the z values at each position of the robot
# z_values = np.array([np.sum([dep.m/(dis**3) for dep in l_deposits]) for dis in dists])
z_values = np.array([np.sum([deposits[i,-1]/(dis[i]**3) + background[i] for i in range(len(deposits))]) for dis in dists])
print(deposits, dists, z_values)
# return z_values, deposits
