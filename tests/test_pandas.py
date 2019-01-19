from pandas import DataFrame
import numpy as np

# d = {'col1': [1, 2], 'col2': [3, 4]}
# mat = DataFrame(data=d)

mat = np.array([[1,0,0], [1,1,0],[1,1,1],[0,0,0]])
mat = DataFrame(data = mat, columns=(["a", "b", "c"]))
print(mat)

print(mat.index)
print(len(mat.index))
print(mat.index[0], mat.index[1], mat.index[2],mat.index[3])
for i in mat.index:
    print(mat["a"].loc[i])
for i in mat.index:
    print(mat.loc[i]["a"])














print("EOF")
