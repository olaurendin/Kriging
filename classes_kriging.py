import sys
sys.path.append("./library_modified")
sys.path.append("./tests")
sys.path.append("./data")

import matplotlib.pyplot as plt
import numpy as np
import os.path
from random import uniform
from scipy.spatial.distance import cdist
from pykrige import ok, uk, uk3d
from geostatsmodels import utilities, geoplot

from classes_extraction import auto_args
from bricks_utilities import extract_points, bricks, smallest_list
import plot_utilities as plut
import ok3d_modified


def background(mu, sigma, scale, l):
    b = scale*np.random.normal(mu, sigma, l)
    return b

def apply_parameters(f, args=False, kwargs=False):
    if args is not False and kwargs is False :
        return f(*args)
    elif args is False and kwargs is not False :
        return f(**kwargs)
    elif args is not False and kwargs is not False :
        return f(*args, **kwargs)
    else:
        raise ValueError("Parameters for the function '{}' incomplete. Please \
check the input list '*args' and/or dictionary parameters '**kwargs' of the function."
        .format(f.__name__))
        sys.exit()

class test_Kriging():

    KRIGING_TYPES = {"ok" : ok.OrdinaryKriging,
                    "ok3d" : ok3d_modified.OrdinaryKriging3D_modified,
                    "uk" : uk.UniversalKriging,
                    "uk3d" : uk3d.UniversalKriging3D}

    @auto_args
    def __init__(self, verbose, session, engine, lims,  msg, imgfolder, img, colx, coly, colz, colpt,
                nb_slice, flg_proj, type_Z_data, params_extraction_db, params_random_deposits,
                params_kriging, params_plots):
        if not img:
            self.img = msg.__class__.__name__
        self.nameimg =  os.path.join(imgfolder, img)

        seq = msg.seq
        self.pos, self.seq, self.lims = extract_points(session, seq, colx, coly, colz, colpt, flg_proj, lims)
        if type_Z_data == "extraction_db":
            print("extraction_db")
            self.data = self.extract_data_db()
        elif type_Z_data == "random_deposits":
            print("random_deposits")
            self.data, self.deposits = self.generate_random_points()
        else:
            print("error")

        self.slice()
        fig, ax = apply_parameters(plut.plotscatterdata, [self.data, self.nameimg], params_plots["plotscatterdata"])
        ax.scatter(self.deposits[:,0], self.deposits[:,1], self.deposits[:,2], c="r", s=16*self.deposits[:,3])
        plt.show()

        fig, mu, std = apply_parameters(plut.plotgaussiandist, [self.data[:,3], self.nameimg], params_plots["plotgaussiandist"])
        plt.show()

        pw = utilities.pairwise(self.pos)
        apply_parameters(plut.laghistogram, [self.data[:,3], self.nameimg, pw],  params_plots["laghistogram"]) #, namefile = imgnamebase )

        kriginginstance = self.train_kriging()

        apply_parameters(plut.plotresidualsvario, [kriginginstance.delta, kriginginstance.sigma, kriginginstance.epsilon, self.nameimg])

        ### TODO : A SUPPRIMER
        n_closest_points = 10

        # self.plotcolormesh(kriginginstance, backend='loop', n_closest_points=n_closest_points)
        apply_parameters(self.plotcolormesh, [kriginginstance], params_plots["plotcolormesh"])
        plt.show()


    def extract_data_db(self):
        ped = self.params_extraction_db
        Z=self.session.query(ped["colZ"]).filter(self.msg.seq.in_(self.seq)).all()
        Z = np.array(Z, dtype=np.float64)
        # Addition of a gaussian noise in the background (optionnal)
        # Calculation of the z values at each position of the robot
        Z += background(ped["mu"], ped["sigma"], ped["scale"])
        return np.concatenate((self.pos, Z.reshape(-1,1)), axis = 1)

    def slice(self):
        if self.nb_slice :
            if self.nb_slice > 1:
                self.data = self.data[::self.nb_slice,:]

    def generate_random_points(self):
        """generate fake magnetic deposits at a given depth. For now every deposit
        has the same magnetical moment but their position is random.

        Parameters
        ----------
        lims : list or tuple, (1,4)
            Limits of the studied brick. The deposits will be created within these borders.
        points : ndarray, (N,3)
            Positions of the robots in 3D.
        avgalt : float
            Fixed altitude (or in this case depth) at which to place the deposits.
        nb_deposits : int, default 10
            Number of fake deposits to generate. The more deposits we generate, the
            closer from a gaussian distribution the distribution of the magnetic levels
            will get, but also the more shadowing effect will take place.

        Returns
        -------
        z_values : ndarray, (1,N)
            The magnetic levels detected at each position of the robot. It is calculated
            as the sum of the influences of all the fake deposits with regard to the
            position of the robot.
        deposits : ndarray, (N,3)
            Positions of the fake deposits generated.
        """
        prd = self.params_random_deposits
        avgz, nb_deposits = prd["avgz"], prd["nb_deposits"]
        new_deposits = prd["new_deposits"]
        magnetical_moment = 1
        xmin, xmax, ymin, ymax, zmin, zmax = self.lims
        if not new_deposits and os.path.exists("./data/deposits.txt"):
            deposits = np.loadtxt("./data/deposits.txt", dtype=np.float64)
        else:
            deposits = np.array([[uniform(xmin,xmax), uniform(ymin, ymax), avgz, magnetical_moment] for i in range(nb_deposits)])
            plut.save_np_array('deposits.txt', deposits, path="./data")
        # Generation of the fake deposits
        if not new_deposits and os.path.exists("./data/deposits.txt"):
            deposits = np.loadtxt("./data/deposits.txt", dtype=np.float64)
        else:
            deposits = np.array([[uniform(xmin,xmax), uniform(ymin, ymax), avgz, magnetical_moment] for i in range(nb_deposits)])
            plut.save_np_array('deposits.txt', deposits, path="./data")
        # Calculation of the distances between each position of the robot and each deposit :
        dists = cdist(self.pos, deposits[:,:3])
        # Addition of a gaussian noise in the background (optionnal)
        b = background(prd["mu"], prd["sigma"], prd["scale"], len(dists))
        # Calculation of the Z values at each position of the robot
        Z = np.array([np.sum([deposits[i,-1]/(dis[i]**3) + b[i] for i in range(len(deposits))]) for dis in dists])
        return np.concatenate((self.pos, Z.reshape(-1,1)), axis = 1), deposits

    def train_kriging(self):
        pk = self.params_kriging
        kriging_function = self.KRIGING_TYPES[pk["kriging_type"]]
        if self.params_kriging["kriging_type"] == "ok":
            print("Ordinary Kriging")
            kr = kriging_function(self.data[:, 0], self.data[:, 1], self.data[:,2], \
            variogram_model = pk["variogram_model"], verbose=self.verbose,
            enable_plotting=pk["enable_plotting"], nlags=pk["nlags"], \
            weight=pk["weight"], enable_statistics = True,\
            anisotropy_scaling = pk["anisotropy_scaling"], anisotropy_angle=pk["anisotropy_angle"])
        elif self.params_kriging["kriging_type"] == "uk":
            print("Universal Kriging")
            kr = kriging_function(self.data[:, 0], self.data[:, 1], self.data[:,2], \
            variogram_model = pk["variogram_model"], verbose=self.verbose,
            enable_plotting=pk["enable_plotting"], nlags=pk["nlags"], weight=pk["weight"],\
            anisotropy_scaling = pk["anisotropy_scaling"], anisotropy_angle=pk["anisotropy_angle"])
        elif self.params_kriging["kriging_type"] in ["ok3d", "uk3d"]:
            print("3D kriging")
            anisotropy_scaling_y, anisotropy_scaling_z = pk["anisotropy_scaling"]
            anisotropy_angle_x, anisotropy_angle_y, anisotropy_angle_z = pk["anisotropy_angle"]
            kr = kriging_function(self.data[:, 0], self.data[:, 1], self.data[:,2], self.data[:,3], \
            variogram_model = pk["variogram_model"], verbose=self.verbose,
            enable_plotting=pk["enable_plotting"], nlags=pk["nlags"], weight=pk["weight"],\
            anisotropy_scaling_y = anisotropy_scaling_y, anisotropy_scaling_z = anisotropy_scaling_z,
            anisotropy_angle_x=anisotropy_angle_x, anisotropy_angle_y=anisotropy_angle_y, anisotropy_angle_z=anisotropy_angle_z)
        else :
            print("Impossible to train kriging : kriging_type '{}' unknown. \
            Available kringing types : {}".format(self.params_kriging["kriging_type"], ["ok", "uk", "ok3d", "uk3d"]))
            sys.exit()
        return kr

    def apply_kriging(self, kriginginstance, style, xpoints, ypoints, zpoints=False, mask=None,
            backend='loop', n_closest_points=None, exact_values=True):
        if self.params_kriging["kriging_type"] in ["ok", "uk"] :
            print("Applying 2D kringing")
            val, ss = kriginginstance.execute(style=style, xpoints= xpoints,
            ypoints=ypoints, mask=mask, backend=backend, n_closest_points=n_closest_points, exact_values=exact_values)
        elif self.params_kriging["kriging_type"] in ["ok3d", "uk3d"] :
            print("Applying 3D kringing")
            val, ss = kriginginstance.execute(style=style, xpoints= xpoints, zpoints = zpoints,
            ypoints=ypoints, mask=mask, backend=backend, n_closest_points=n_closest_points, exact_values=exact_values)
        else :
            print("Impossible to apply kriging : kriging_type '{}' unknown. \
            Available kringing types : {}".format(self.params_kriging["kriging_type"], ["ok", "uk", "ok3d", "uk3d"]))
            sys.exit()
        return val, ss

    # def limsplotcolormesh():
    #     if self.params_random_deposits is not False :
    #         lims = smallest_list(self.lims, self.params_random_deposits["lims"])
    #         lims = np.minimum(lims, self.)
    #     return lims, avgalt

    def plotcolormesh(self, kriginginstance, xpoints=False, ypoints=False, zpoints=False, backend="loop", n_closest_points=10):
        # lims, avgz = lims_pclm = limsplotcolormesh()
        xmin, xmax, ymin, ymax = self.lims[:4]
        print(xmin, xmax, ymin, ymax)
        prec = 100
        gridx = np.linspace(xmin, xmax, prec)
        gridy = np.linspace(ymin, ymax, prec)
        if self.type_Z_data == "random_deposits":
            if self.params_random_deposits is False :
                raise ValueError("Must specify params_random_deposits when type_Z_data == random_deposits")
            else :
                avgz = self.params_random_deposits["avgz"]
                gridz = avgz*np.ones(gridx.shape)
                val, ss = self.apply_kriging(kriginginstance,'grid', gridx, gridy, gridz, backend = backend, n_closest_points=n_closest_points)
                fig, ax0, ax1 = plut.plotcolormesh_estvar_fakedeposits(self.data,self.deposits,gridx,gridy,
                val[0,...], ss[0,...], nbins = 40,typeplot="contourf", namefile = "test")
        elif self.params_kriging["kriging_type"] in ["ok3d", "uk3d"]:
            avgz = self.params_random_deposits["avgz"]
            gridz = avgz*np.ones(gridx.shape)
            val, ss = self.apply_kriging(kriginginstance,'grid', gridx, gridy, gridz, backend = backend, n_closest_points=n_closest_points)
            fig, ax0, ax1 = plut.plotcolormesh_estvar(self.data,gridx,gridy,
                    val[0,...], ss[0,...], nbins = 40,typeplot="contourf", namefile = "test")
            # val, ss = kriginginstance.execute('grid', gridx, gridy, avgalt, exact_values=True)
        elif self.params_kriging["kriging_type"] in ["ok", "uk"]:
            val, ss = self.apply_kriging(kriginginstance,'grid', gridx, gridy, backend = backend, n_closest_points=n_closest_points)
            fig, ax0, ax1 = plut.plotcolormesh_estvar(self.data,gridx,gridy,
                    val[0,...], ss[0,...], nbins = 40,typeplot="contourf", namefile = "test")
        else :
            print("Impossible to plot colormesh : kriging_type '{}' unknown. \
            Available kringing types : {}".format(self.params_kriging["kriging_type"], ["ok", "uk", "ok3d", "uk3d"]))
            sys.exit()
        return fig, ax0, ax1






















# "EOF"
