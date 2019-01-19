import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

P = np.array([[0,0,1],
            [0,1,0],
            [1,0,0]])
y=np.array([[0,0,2]])
print(squareform(pdist(P)))
print(cdist(P,y))
m_values = np.array([[(i*j) for i in range(10)] for j in range(10)])
print(np.sum(m_values, axis = 1))
