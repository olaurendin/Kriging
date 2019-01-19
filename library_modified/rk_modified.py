# coding: utf-8
from pykrige.compat import validate_sklearn
validate_sklearn()
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from pykrige.rk import *

class Krige_modified(Krige):

    def __init__(self,
                 method='ordinary',
                 variogram_model='linear',
                 nlags=6,
                 weight=False,
                 n_closest_points=10,
                 verbose=False,
                 variogram_parameters=None):

        validate_method(method)
        self.variogram_model = variogram_model
        self.verbose = verbose
        self.nlags = nlags
        self.weight = weight
        self.model = None  # not trained
        self.n_closest_points = n_closest_points
        self.method = method
        self.variogram_parameters = variogram_parameters

    def fit(self, x, y, *args, **kwargs):
        """
        Parameters
        ----------
        x: ndarray
            array of Points, (x, y) pairs of shape (N, 2) for 2d kriging
            array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
        y: ndarray
            array of targets (N, )
        """

        points = self._dimensionality_check(x)

        # if condition required to address backward compatibility
        if self.method in threed_krige:
            self.model = krige_methods[self.method](
                val=y,
                variogram_model=self.variogram_model,
                nlags=self.nlags,
                weight=self.weight,
                verbose=self.verbose,
                variogram_parameters=self.variogram_parameters,
                **points
            )
        else:
            self.model = krige_methods[self.method](
                z=y,
                variogram_model=self.variogram_model,
                nlags=self.nlags,
                weight=self.weight,
                verbose=self.verbose,
                variogram_parameters=self.variogram_parameters,
                **points
            )
