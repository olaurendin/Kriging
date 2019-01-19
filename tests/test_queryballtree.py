import numpy as np
import scipy.spatial as sp

X = np.array([[0,0,0],
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1,1,0],
            [1,0,1],
            [1,1,1],
            [0,1,1],
            [0.5,0.5,0.5]])

eff_range = 1
tree = sp.cKDTree(X)
neighbors = tree.query_ball_tree(tree,eff_range)

for i in range(len(neighbors)):
    print(neighbors[i])
    neighbors[i].remove(i)
    print(neighbors[i])

l = [i for i in range(10)]
l.remove(0)
print(l)
