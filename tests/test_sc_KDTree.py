from scipy.spatial import cKDTree
import numpy as np

data=np.array([[0,0,1],
                [0,1,0],
                [1,0,0],
                [0,0,2]])
n_closest_points=2
tree = cKDTree(data)
eps = 1.e-10   # Cutoff for comparison to zero
bd, bd_idx = tree.query(data, k=n_closest_points, eps=eps)
print(bd, bd_idx)
r=1
res = tree.query_ball_point(data, r,eps=eps)
print(res)
res2 = tree.query_ball_tree(tree,r,eps=eps)
print(res2)
res3 = tree.sparse_distance_matrix(tree, r)
print(res3, res3[0,0])


l = [1]
print(l.remove(0))
