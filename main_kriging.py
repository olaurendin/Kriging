from classes_sqlalchemyORM import *
from classes_kriging import test_Kriging
from bricks_utilities import bricks
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import numpy as np

"""
################################################################################
                            GOAL OF THIS SCRIPT
################################################################################

This script aims at choosing kriging parameters to apply on the data previously
uploaded in the db. After selecting how to extract the training points Coordinates
from the db, you can either choose
"""

"""
################################################################################
INITIALISATION :
Set main variables for the database.
"""

### Connection parameters to the database
dialect = "postgresql"
driver = "psycopg2"
user = "ganlan"
password = "password"
host = "localhost"
dbname = "dataAUV"
verbose = True

### dialect[+driver]://user:password@host/dbname[?key=value..]
### doc : http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine
engine = create_engine("{}+{}://{}:{}@{}/{}".format(dialect, driver, user, password, host, dbname), echo=False)
### engine = create_engine('postgresql+psycopg2://ganlan:password@localhost/dataAUV', echo=True)
### Connection to the database
Session = sessionmaker(bind=engine)
session = Session()

"""
################################################################################
KRIGING :
"""

test_Kriging(
verbose = verbose,
session = session,
engine = engine,


lims = bricks["11"],               # Brick of coordinates to be extracted from db
msg = ROSMsg_fix_137,              # Message selected, and thus the database table
imgfolder = "../redaction/photos", # Folder where images of plots will be stored
img = "ROSMsg_fix_137"+"_b11", # General pattern of the images' name.

### DB columns corresponding to points coordinates. Can be 2D or 3D
colx = ROSMsg_fix_137.lon,      # Coordinates of the points w.r.t x
coly = ROSMsg_fix_137.lat,      # Coordinates of the points w.r.t y
colz = ROSMsg_fix_137.alt,      # Coordinates of the points w.r.t z  (can be put to False if 2D problem)
colpt = ROSMsg_fix_137.pt,      # Coordinates of the GeoIndex to speed up extraction of bricks of points

nb_slice = 10,            # Extraction of lines out of the dataset. 1 data out of every slice is extracted

flg_proj = False,     # If the lims are not in the same projection than the values of colx, coly, colz

### Selection of type of extraction : Choose one of the following :
#       extraction_db <=> Coordinates and magnetical data are directly taken from DB
#       random_deposits <=> Coordinates are taken from DB but magnetical data is simulated at each coordinate point from fake iron deposits
# type_Z_data = "extraction_db",
type_Z_data = "random_deposits",

### if type_data = "extraction_db" :
#       The message/column from rosbag/TXT file representing the data to be kriged must be provided
params_extraction_db = {
        "colZ" : ROSMsg_fix_137.alt,    # The message/column from rosbag/TXT file representing the data to be kriged
        "mu" : 1,                       # mean of the gaussian additive noise (optionnal)
        "sigma" : 0.25,                 # standard deviation of the gaussian additive noise (optionnal)
        "scale" : 1,},


### elif type_data = "random_deposits" :
#       The fake data to be kriged must be simulated :
params_random_deposits = {"new_deposits":False,     # If False, uses previously generated fake data
                         "lims" : False,
                         "avgz" : 0,                # Mean of the fake data
                         "nb_deposits" : 300,       # Number of fake iron deposits to simulate the magnetical field
                         "mu" : 1,                  # mean of the gaussian additive noise (optionnal)
                         "sigma" : 1,               # standard deviation of the gaussian additive noise (optionnal)
                         "scale" : 0},

### Parameters of the kriging algorithm :
params_kriging={"kriging_type" : "ok3d",            # Type of kriging, choose from "ok", "uk" (2D) or "ok3d", "uk3d" (3D)
                "variogram_model" : "spherical",    # Variogram model
                "enable_plotting" : True,
                "nlags" : 15,                       # Number of lags
                "weight" : True,                    # If True, gives greater importance to small distances lags.
                "anisotropy_scaling" : [1.0, 1.0],      # Scale of anisotropy (ignored if anisotropy angle =  [0.0, 0.0, 0.0])
                "anisotropy_angle" : [0.0, 0.0, 0.0]},  # Angle of anisotropy, put to [0.0, 0.0, 0.0] if no anisotropy in your model.

### Plots parameters : all plots whose name appear in the dictionnary keys will be plotted, comment these otherwise.
### Look in plot_utilities.py for more details of these functions.
params_plots = {
                # Plot the extracted data points in 2D / 3D
                "plotscatterdata" : { "cmap":"jet", "xlabel":"E-W axis[m]", "ylabel":"N-S axis[m]",
                "zlabel":"altitude [m]", "cblabel":"Measurements"},

                # Plot distribution of distances w.r.t lags and compared to their closest gaussian distributiuon
                "plotgaussiandist" : {"bins":15, "xlabel":'Magnetical Data [T]'},

                # Plot lags population as an histogram
                "laghistogram" : {"lags":np.linspace(0, 1500, 30), "tol":30 , "width":30},

                # Plots the residuals after kriging (= training accuracy)
                "plotresidualsvario" : {},

                # Plots the results of kriging at each testing point.
                "plotcolormesh" : {"backend":'loop', "n_closest_points":10}
                }

)


print("EOF")
