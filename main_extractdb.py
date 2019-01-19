# "Standard" libraries
import sys
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# folders of interest
sys.path.append("./library_modified")
sys.path.append("./tests")
sys.path.append("./data")
sys.path.append("./logs")

# Pykrige classes modified
import ok3d_modified as ok3d

# Additional home made libraries
from bricks_utilities import extract_points, bricks
import plot_utilities as plut
from classes_extraction import Ext_msgs_db
from classes_sqlalchemyORM import *
from proj_utilities import *

"""
################################################################################
                            GOAL OF THIS SCRIPT
################################################################################

This script aims at create a database containing all the data later used in the
kriging algorithm. The coordinates of the training points can be taken from
different files format : rosbag or CSV/TXT files. The magnetical data can be
either extracted from a file of simulated using fake iron deposits on the
seafloor. Finally, this script provides the ability to change the coordinate
system of the coordinates of the training points before uploading them to the DB.

Before using this script, you must provide the database structure by creating a
SQLAlchemy class in the classes_sqlalchemyORM.py file and the input and output
projections you need in the proj_utilities.py file (more info on this later on
in the script).

Due to an unsolved memory leak, the upload of files to the DB must be done one
file at a time.
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
verbose = False

### dialect[+driver]://user:password@host/dbname[?key=value..]
### doc : http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine
engine = create_engine("{}+{}://{}:{}@{}/{}".format(dialect, driver, user, password, host, dbname), echo=verbose)
### engine = create_engine('postgresql+psycopg2://ganlan:password@localhost/dataAUV', echo=True)
### Connection to the database
Session = sessionmaker(bind=engine)
session = Session()

"""
################################################################################
EXTRACTION MESSAGES :
Choose a set of messages to extract and upload into database for further usage.
"""


msg_to_db = Ext_msgs_db(
engine = engine,
session = session,

### Extraction parameters : which messages to upload to db
type_extraction = "file_to_db",      # The msgs to extract are in a CSV/TXT/ROSBAG file
# type_extraction = "create_cropped_file",      # Creates a cropped CSV/rosbag file for further usage
# type_extraction = "random_deposits",   # Generate random points for testing

### If type_extraction == "file_to_db" :
### data file where the data to store in database is located.
namefile = "cropped10000rosbag_2018-07-11-10-23-16_0.bag",   # Name of the file where the data is saved. Must be given with its extension. Supported extensions : .txt, .csv, .bag
pathfile = "./data/Boatbot-exp/117/boatbot_zotac117/2018-07-11-10:23:05/cropped",        # path where the file is situated
# namefile = "data_nav.txt",
# pathfile = "/home/ganlan/internship/Final_code/data",

### If the file to extract data from is a CSV / TXT file :
params_file_extract_csv = {"msg_class" : Msg_nav_data,          # class described in classes_sqlalchemyORM for the db structure
                        "sep" : " ",                            # Separator in the CSV file
                        "nb_lines": 100,                        # Nb of lines to extract from CSV (= np.inf if you want all the lines)
                        "header" : 0,                           # nb of lines the header of the CSV file takes
                        "columns":{"time":"t",                  # Link between the names of the columns in DB described in the SQLAlchemy class
                                    "x":"x",                    #   in classes_sqlalchemyORM.py and their respective name in the CSV file.
                                    "y":"y",                    #   "SQLAlchemy class column name" : "CSV column name"
                                    "depth":"depth",
                                    "a":"a"},
                        "extra":["depth_seabed = depth+a",      # In case you want to create a DB column as a combination of several CSV column,
                                "pt = input_pt(x,y)"]           #   you can define it here as a string equation. Keep in mind that every non numerical term
                                                                #   appearing in the equation must be define beforehand in the "columns" attribute.
                                                                #   You can use custom functions by defining them in a separate .py file and importing
                                                                #   all of it (*) in the classes_extration.py file.
                                                                #   Look at input_pt defined in proj_utilities.py for an example.
                                                                #   Please input an empty list if no extra column is needed.
                                                                #   The function input_pt creates a Geoindex column.
                                                                #   The function input_pt_proj creates a Geoindex column projected into a new system
                                                                #   of coordinates described in proj_utilities.py
                        },

### If the file to extract data from is a rosbag :
###     As several topics can be extracted from a rosbag, several tables will be
###     created in the DB, one for each topic. The keys of the following dictionnary
###     are the complete name of the ros topics to extract. The value of the dictionnary
###     for each key is structured in the same manner than for CSV files described above.
params_file_extract_bag = { "/zodiac_auto/fix" : { "msg_class" : ROSMsg_fix_117_test,
                                                    "nb_msgs" : np.inf,
                                                    "columns" : {"time_sec" : "header.stamp.secs",
                                                                "time_nsec" : "header.stamp.nsecs",
                                                                "frame_id" : "header.frame_id",
                                                                "lat" : "latitude",
                                                                "lon" : "longitude",
                                                                "alt" : "altitude"},
                                                    "extra" : ["pt = input_pt_proj(lon, lat)"]},
                        # "/zodiac_auto/gps_course" : {"msg_class" : ROSMsg_test2,
                        #                             "nb_msgs" : np.inf,
                        #                             "columns" : {"data" : "data"},
                        #                             "extra" : []}
            },

### If you use input_pt_proj, it means that you apply a projection of coordinates before
### inputing your data to the DB. The default projection is from standard GPS coordinates towards Lambert93.
### If you wish to change this setting, please go to proj_utilities.py
### and modify the input and output projections (epsg_inproj and epsg_outproj)
### Use the epsg number which corresponds to the projections you are looking for :
### See http://spatialreference.org/ref/epsg/

### If type_extraction == "create_cropped_file" :
### The paramaters taken are those of params_file_extract_bag or params_file_extract_csv
### Depending on the original file type.
nb_msgs_cropped_file = 100,
path_cropped_file = "./data",

### If type_extraction == "random_deposits" (3D coordinates points only) :
params_random_deposits = { "lims" : bricks["11"],
                            "seq" : ROSMsg_fix_117_test.seq,
                            "colx" : ROSMsg_fix_117_test.lon,
                            "coly" : ROSMsg_fix_117_test.lat,
                            "colz" : ROSMsg_fix_117_test.alt,
                            "colpt" : ROSMsg_fix_117_test.pt,
                            "flg_proj" : False,      # If the lims are not in the same projection than the values of colx, coly, colz
                            # "points" : ,
                            "new_deposits" : True,   # If False, takes previously generated fake deposits.
                            "avgz" : 7,
                            "nb_deposits" : 10,
                            "mu" : 1,
                            "sigma" : 1,
                            "scale" : 0},

### type_insertion : parameters to extract the messages in the database
type_insertion = "concatenate_table",      # Add the extracted msgs to an already uploaded db of the same name.
# type_insertion = "replace_table",        # Erases the content of the selected db before adding elements

# TODO : verbose
verbose = False,      # To activate multiple prints during the extraction of the messages

# TODO : logfile ; cropped file
### logfile : txt file to save the logs.
flg_autolog = True,  # If True, the name of the logfile is generated automatically from the namefile.
flg_newlog = True,   # If True, the new logfile replaces would be older ones
namelogfile = "logfile.txt",
pathlogfile = "./logs",

)

print("EOF")
