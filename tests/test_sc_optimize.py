import numpy as np
import scipy as sc
from pykrige.variogram_models import *

thresh=0.99
vmp = [50, 100, 5]

if isinstance(vmp, list):
    # v = [psill, range, nugget]
    v = np.array([[vmp[0]], [vmp[1]], [vmp[2]]])
    print(vmp, v)
    eff_range = sc.optimize.fsolve(lambda x: gaussian_variogram_model(v, x) - thresh*(vmp[0]+vmp[2]), vmp[1])
    print(eff_range)
    print(gaussian_variogram_model(v, eff_range), thresh*(vmp[0]+vmp[2]))
