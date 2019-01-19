import numpy as np



x = np.arange(10)
y = np.arange(10)
np.random.seed(42)
np.random.shuffle(x)
np.random.seed(42)
np.random.shuffle(y)
print(x,y)

l = []
l2={}
print(isinstance(l,list))
print(isinstance(l2,dict))
