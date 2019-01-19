from pykrige.ok3d import *
from pykrige.core import _adjust_for_anisotropy, _initialize_variogram_model, \
    _make_variogram_parameter_list

from core_modified import _find_statistics

class OrdinaryKriging3D_modified(OrdinaryKriging3D):

    def __init__(self, x, y, z, val, variogram_model='linear',
                 variogram_parameters=None, variogram_function=None, nlags=6,
                 weight=False, anisotropy_scaling_y=1., anisotropy_scaling_z=1.,
                 anisotropy_angle_x=0., anisotropy_angle_y=0.,
                 anisotropy_angle_z=0., verbose=False, enable_plotting=False):

        # Code assumes 1D input arrays. Ensures that any extraneous dimensions
        # don't get in the way. Copies are created to avoid any problems with
        # referencing the original passed arguments.
        self.X_ORIG = \
            np.atleast_1d(np.squeeze(np.array(x, copy=True, dtype=np.float64)))
        self.Y_ORIG = \
            np.atleast_1d(np.squeeze(np.array(y, copy=True, dtype=np.float64)))
        self.Z_ORIG = \
            np.atleast_1d(np.squeeze(np.array(z, copy=True, dtype=np.float64)))
        self.VALUES = \
            np.atleast_1d(np.squeeze(np.array(val, copy=True, dtype=np.float64)))

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG))/2.0
        self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG))/2.0
        self.ZCENTER = (np.amax(self.Z_ORIG) + np.amin(self.Z_ORIG))/2.0
        self.anisotropy_scaling_y = anisotropy_scaling_y
        self.anisotropy_scaling_z = anisotropy_scaling_z
        self.anisotropy_angle_x = anisotropy_angle_x
        self.anisotropy_angle_y = anisotropy_angle_y
        self.anisotropy_angle_z = anisotropy_angle_z
        if self.verbose:
            print("Adjusting data for anisotropy...")
        self.X_ADJUSTED, self.Y_ADJUSTED, self.Z_ADJUSTED = \
            _adjust_for_anisotropy(np.vstack((self.X_ORIG, self.Y_ORIG, self.Z_ORIG)).T,
                                   [self.XCENTER, self.YCENTER, self.ZCENTER],
                                   [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
                                   [self.anisotropy_angle_x, self.anisotropy_angle_y, self.anisotropy_angle_z]).T

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        if self.variogram_model not in self.variogram_dict.keys() and self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for "
                                 "custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        if self.verbose:
            print("Initializing variogram model...")

        vp_temp = _make_variogram_parameter_list(self.variogram_model,
                                                 variogram_parameters)
        self.lags, self.semivariance, self.variogram_model_parameters = \
            _initialize_variogram_model(np.vstack((self.X_ADJUSTED,
                                                   self.Y_ADJUSTED,
                                                   self.Z_ADJUSTED)).T,
                                        self.VALUES, self.variogram_model,
                                        vp_temp, self.variogram_function,
                                        nlags, weight, 'euclidean')

        if self.verbose:
            if self.variogram_model == 'linear':
                print("Using '%s' Variogram Model" % 'linear')
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], '\n')
            elif self.variogram_model == 'power':
                print("Using '%s' Variogram Model" % 'power')
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
            elif self.variogram_model == 'custom':
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print("Full Sill:", self.variogram_model_parameters[0] +
                      self.variogram_model_parameters[2])
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        self.delta, self.sigma, self.epsilon = \
            _find_statistics(np.vstack((self.X_ADJUSTED,
                                        self.Y_ADJUSTED,
                                        self.Z_ADJUSTED)).T,
                             self.VALUES,  self.variogram_function,
                             self.variogram_model_parameters, 'euclidean')
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print("Q1 =", self.Q1)
            print("Q2 =", self.Q2)
            print("cR =", self.cR, '\n')

    def execute(self, style, xpoints, ypoints, zpoints, mask=None,
                backend='vectorized', n_closest_points=None, exact_values=True):
        """Calculates a kriged grid and the associated variance.

        This is now the method that performs the main kriging calculation.
        Note that currently measurements (i.e., z values) are
        considered 'exact'. This means that, when a specified coordinate
        for interpolation is exactly the same as one of the data points,
        the variogram evaluated at the point is forced to be zero.
        Also, the diagonal of the kriging matrix is also always forced
        to be zero. In forcing the variogram evaluated at data points
        to be zero, we are effectively saying that there is no variance
        at that point (no uncertainty, so the value is 'exact').

        In the future, the code may include an extra 'exact_values' boolean
        flag that can be adjusted to specify whether to treat the
        measurements as 'exact'. Setting the flag to false would indicate
        that the variogram should not be forced to be zero at zero distance
        (i.e., when evaluated at data points). Instead, the uncertainty in the
        point will be equal to the nugget. This would mean that the diagonal
        of the kriging matrix would be set to the nugget instead of to zero.

        Parameters
        ----------
        style : str
            Specifies how to treat input kriging points.
            Specifying 'grid' treats xpoints, ypoints, and zpoints as arrays of
            x, y, and z coordinates that define a rectangular grid.
            Specifying 'points' treats xpoints, ypoints, and zpoints as arrays
            that provide coordinates at which to solve the kriging system.
            Specifying 'masked' treats xpoints, ypoints, and zpoints as arrays
            of x, y, and z coordinates that define a rectangular grid and uses
            mask to only evaluate specific points in the grid.
        xpoints : array_like, shape (N,) or (N, 1)
            If style is specific as 'grid' or 'masked', x-coordinates of
            LxMxN grid. If style is specified as 'points', x-coordinates of
            specific points at which to solve kriging system.
        ypoints : array-like, shape (M,) or (M, 1)
            If style is specified as 'grid' or 'masked', y-coordinates of
            LxMxN grid. If style is specified as 'points', y-coordinates of
            specific points at which to solve kriging system.
            Note that in this case, xpoints, ypoints, and zpoints must have the
            same dimensions (i.e., L = M = N).
        zpoints : array-like, shape (L,) or (L, 1)
            If style is specified as 'grid' or 'masked', z-coordinates of
            LxMxN grid. If style is specified as 'points', z-coordinates of
            specific points at which to solve kriging system.
            Note that in this case, xpoints, ypoints, and zpoints must have the
            same dimensions (i.e., L = M = N).
        mask : boolean array, shape (L, M, N), optional
            Specifies the points in the rectangular grid defined by xpoints,
            ypoints, zpoints that are to be excluded in the
            kriging calculations. Must be provided if style is specified
            as 'masked'. False indicates that the point should not be masked,
            so the kriging system will be solved at the point.
            True indicates that the point should be masked, so the kriging
            system should will not be solved at the point.
        backend : str, optional
            Specifies which approach to use in kriging. Specifying 'vectorized'
            will solve the entire kriging problem at once in a
            vectorized operation. This approach is faster but also can consume a
            significant amount of memory for large grids and/or large datasets.
            Specifying 'loop' will loop through each point at which the kriging
            system is to be solved. This approach is slower but also less
            memory-intensive. Default is 'vectorized'.
        n_closest_points : int, optional
            For kriging with a moving window, specifies the number of nearby
            points to use in the calculation. This can speed up the calculation
            for large datasets, but should be used with caution.
            As Kitanidis notes, kriging with a moving window can produce
            unexpected oddities if the variogram model is not carefully chosen.
        exact_values : boolean, optional
            Boolean flag that can be adjusted to specify whether to treat the
            measurements as 'exact'. Setting the flag to false would indicate
            that the variogram should not be forced to be zero at zero distance
            (i.e., when evaluated at data points). Instead, the uncertainty in the
            point will be equal to the nugget. This would mean that the diagonal
            of the kriging matrix would be set to the nugget instead of to zero.

        Returns
        -------
        kvalues : ndarray, shape (L, M, N) or (N, 1)
            Interpolated values of specified grid or at the specified set
            of points. If style was specified as 'masked', kvalues will be a
            numpy masked array.
        sigmasq : ndarray, shape (L, M, N) or (N, 1)
            Variance at specified grid points or at the specified set of points.
            If style was specified as 'masked', sigmasq will be a numpy
            masked array.
        """

        if self.verbose:
            print("Executing Ordinary Kriging...\n")

        if style != 'grid' and style != 'masked' and style != 'points':
            raise ValueError("style argument must be 'grid', 'points', "
                             "or 'masked'")

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        zpts = np.atleast_1d(np.squeeze(np.array(zpoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        nz = zpts.size
        a = self._get_kriging_matrix(n, exact_values)

        if style in ['grid', 'masked']:
            if style == 'masked':
                if mask is None:
                    raise IOError("Must specify boolean masking array when "
                                  "style is 'masked'.")
                if mask.ndim != 3:
                    raise ValueError("Mask is not three-dimensional.")
                if mask.shape[0] != nz or mask.shape[1] != ny or mask.shape[2] != nx:
                    if mask.shape[0] == nx and mask.shape[2] == nz and mask.shape[1] == ny:
                        mask = mask.swapaxes(0, 2)
                    else:
                        raise ValueError("Mask dimensions do not match "
                                         "specified grid dimensions.")
                mask = mask.flatten()
            npt = nz * ny * nx
            grid_z, grid_y, grid_x = np.meshgrid(zpts, ypts, xpts, indexing='ij')
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()
            zpts = grid_z.flatten()
        elif style == 'points':
            if xpts.size != ypts.size and ypts.size != zpts.size:
                raise ValueError("xpoints, ypoints, and zpoints must have "
                                 "same dimensions when treated as listing "
                                 "discrete points.")
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', "
                             "'points', or 'masked'")

        xpts, ypts, zpts = \
            _adjust_for_anisotropy(np.vstack((xpts, ypts, zpts)).T,
                                   [self.XCENTER, self.YCENTER, self.ZCENTER],
                                   [self.anisotropy_scaling_y, self.anisotropy_scaling_z],
                                   [self.anisotropy_angle_x, self.anisotropy_angle_y, self.anisotropy_angle_z]).T

        if style != 'masked':
            mask = np.zeros(npt, dtype='bool')

        xyz_points = np.concatenate((zpts[:, np.newaxis], ypts[:, np.newaxis],
                                     xpts[:, np.newaxis]), axis=1)
        xyz_data = np.concatenate((self.Z_ADJUSTED[:, np.newaxis],
                                   self.Y_ADJUSTED[:, np.newaxis],
                                   self.X_ADJUSTED[:, np.newaxis]), axis=1)

        if n_closest_points is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(xyz_data)
            print("tree ok")
            bd, bd_idx = tree.query(xyz_points, k=n_closest_points, eps=0.0)
            if backend == 'loop':
                kvalues, sigmasq = \
                    self._exec_loop_moving_window(a, bd, mask, bd_idx)
                print("moving_window ok")
            else:
                raise ValueError("Specified backend '{}' not supported "
                                 "for moving window.".format(backend))
        else:
            bd = cdist(xyz_points, xyz_data, 'euclidean')
            if backend == 'vectorized':
                kvalues, sigmasq = self._exec_vector(a, bd, mask)
            elif backend == 'loop':
                kvalues, sigmasq = self._exec_loop(a, bd, mask)
            else:
                raise ValueError('Specified backend {} is not supported for '
                                 '3D ordinary kriging.'.format(backend))

        if style == 'masked':
            kvalues = np.ma.array(kvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ['masked', 'grid']:
            kvalues = kvalues.reshape((nz, ny, nx))
            sigmasq = sigmasq.reshape((nz, ny, nx))

        return kvalues, sigmasq

    def _get_kriging_matrix(self, n, exact_values):
        """Assembles the kriging matrix."""

        xyz = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                              self.Y_ADJUSTED[:, np.newaxis],
                              self.Z_ADJUSTED[:, np.newaxis]), axis=1)
        d = cdist(xyz, xyz, 'euclidean')
        a = np.zeros((n+1, n+1))
        a[:n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
        if not exact_values:
            if self.variogram_model == 'linear':
                np.fill_diagonal(a, self.variogram_model_parameters[1])
            elif self.variogram_model != 'custom':
                np.fill_diagonal(a, self.variogram_model_parameters[2])
        else :
            np.fill_diagonal(a, 0.)
        a[n, :-1] = 1.0
        a[:-1, n] = 1.0

        return a

    def _exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= self.eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= self.eps)

        b = np.zeros((npt, n+1, 1))
        b[:, :n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(mask[:, np.newaxis, np.newaxis], n+1, axis=1)
            b = np.ma.array(b, mask=mask_b)

        x = np.dot(a_inv, b.reshape((npt, n+1)).T).reshape((1, n+1, npt)).T
        kvalues = np.sum(x[:, :n, 0] * self.VALUES, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return kvalues, sigmasq
