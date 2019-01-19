from scipy.spatial.distance import cdist
import numpy as np
import scipy as sc
import sys

# X = np.zeros((10,3))
# X[:,0]=np.arange(1,11)
# print(X)
# c = cdist(X, X, 'euclidean')
# print(c)
# l = int(X.shape[0]/2)
# print(l)
# print(X[:l], X[l:])
# c00=cdist(X[:l], X[:l], 'euclidean')
# c01=cdist(X[:l], X[l:], 'euclidean')
# # c10=cdist(X[l:, :l], X[l:, :l], 'euclidean')
# c11=cdist(X[l:], X[l:], 'euclidean')
# c2=np.block([[c00,c01], [c01,c11]])
# print(c00,c01,c11,c2)


def rec_cdist(X, Y, metric = "euclidean"):
    try :
        c = cdist(X, Y, metric=metric)
    except ValueError :
        print("The shape of the matrix is too big to be calculated by cdist directly,\
        subdivision of the matrix initiated. It might take a moment")
        l = int(X.shape[0]/2)
        print("########",l,"\n")
        c00=rec_cdist(X[:l], Y[:l], metric=metric)
        c01=rec_cdist(X[:l], Y[l:], metric=metric)
        c11=rec_cdist(X[l:], Y[l:], metric=metric)
        c=np.block([[c00,c01], [c01,c11]])
    return c



if __name__ == "__main__":
    # len = 100000
    # X = np.zeros((len,3), dtype=np.float64)
    # X[:,0]=np.arange(0,len)
    # X.astype("float32")
    # print(rec_cdist(X,X))
    print(np.intp)
    print(sys.maxsize > 2**32)
    print(sys.executable)
