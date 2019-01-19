from inspect import signature
import os
import types
import time
import rosbag
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from pandas import DataFrame
import pandas as pd
import numpy as np
from random import uniform
from scipy.spatial.distance import cdist
from tqdm import tqdm

from classes_sqlalchemyORM import *
from proj_utilities import *
import proj_utilities
from bricks_utilities import extract_points, extract_points_old

def auto_args(f):
    """
    Decorator for auto assignation of arguments during class instanciation.
    Code by judy2k from StackOverflow :
    https://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
    """
    sig = signature(f)  # Get a signature object for the target:
    def replacement(self, *args, **kwargs):
        # Parse the provided arguments using the target's signature:
        bound_args = sig.bind(self, *args, **kwargs)
        # Save away the arguments on `self`:
        for k, v in bound_args.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        # Call the actual constructor for anything else:
        f(self, *args, **kwargs)
    return replacement

def getattr_recur(obj, elmt):
    """ Recursive version of getattr """
    elmts = elmt.split(".")
    if len(elmts)==1:
        return getattr(obj, elmts[0])
    else:
        return getattr_recur(getattr(obj, elmts[0]), ".".join(elmts[1:]))

def set_attributes_from_ROSMsg(msg_db, msg, dict_corresp):
    """ Does the connexion between a rosmsg and an sqlalchemy instance except geoindexes """
    for i in dict_corresp.items():
        setattr(msg_db,i[0],getattr_recur(msg,i[1]))
    return msg_db

def set_attributes_from_csv(msg_db, msg, dict_corresp, ind):
    """ Does the connexion between a csv and an sqlalchemy instance except geoindexes """
    for i in dict_corresp.items():
        setattr(msg_db,i[0],msg[i[1]].loc[ind])
    return msg_db

def set_extra(msg_db, extra):
    dict_columns = msg_db.__dict__
    l_func_proj_utilities = [getattr(proj_utilities, a) for a in dir(proj_utilities)
      if isinstance(getattr(proj_utilities, a), types.FunctionType)]
    for f in l_func_proj_utilities:
        dict_columns[f.__name__] = f
    for ext in extra:
        setcol, op = ext.replace(" ","").split("=")
        setattr(msg_db, setcol, eval(op, dict_columns))
    return msg_db

def get_delta_seq(session, rosmsg_class, header_seq):
    try :
        maxseq = session.query(func.max(rosmsg_class.seq)).one()[0]
        if (maxseq > header_seq):
            delta_seq = maxseq
        else:
            delta_seq=0
    except TypeError:
        delta_seq = 0
    return delta_seq


class Ext_msgs_db():

    @auto_args
    def __init__(self, engine, session, type_extraction = "file_to_db", params_file_extract_csv = {},
        params_file_extract_bag = {}, namefile = "data_nav.txt", pathfile = "./data", epsg_inproj = 'epsg:4326',
        epsg_outproj = 'epsg:2154', nb_msgs_cropped_file=100, path_cropped_file = "./data",
        params_random_deposits = {}, type_insertion = "concatenate_table", verbose = True,
        flg_autolog = True, flg_newlog = True, namelogfile = "logfile.txt", pathlogfile = "./logs"):

        self.init_table_db()

        if type_extraction == "file_to_db":
            self.file = os.path.join(self.pathfile, self.namefile)
            if self.namefile[-4:] in [".txt", ".csv"]:
                self.extract_csv()
            elif self.namefile[-4:] == ".bag":
                self.extract_rosbag()
            else :
                print("File format {} not recogized, formats recognized :{}"
                    .format(self.namefile[-4:], [".txt", ".csv", ".bag"]))
        elif type_extraction == "create_cropped_file":
            # self.file = os.path.join(self.pathfile, self.namefile)
            if self.namefile[-4:] in [".txt", ".csv"]:
                self.create_cropped_csv(namefile, pathfile, path_cropped_file, nb_msgs_cropped_file)
            elif self.namefile[-4:] == ".bag":
                self.create_cropped_bag(namefile, pathfile, path_cropped_file, nb_msgs_cropped_file)
            else :
                print("File format {} not recogized, formats recognized :{}"
                    .format(self.namefile[-4:], [".txt", ".csv", ".bag"]))
        elif type_extraction == "random_deposits":
            self.generate_random_points()

    def extract_rosbag(self):
        """ Send querries to the database to add the ros messages to the designed db table. """

        print("Upload data to database {} initiated. This may take a while.".format(
        rosmsg_class.__tablename__))
        with rosbag.Bag(self.file, 'r') as bag:
            print("rosbag {} opened".format(bag.filename))
            bagName = bag.filename
            topics = list(bag.get_type_and_topic_info()[1].keys())
            msgs = list(bag.get_type_and_topic_info()[1].values())
            types = [msgs[i].msg_type for i in range(len(msgs))]

            topicstoext = self.params_file_extract_bag.keys()

            topics = [topic for topic in topics if topic in topicstoext]
            nmsgmax = 0
            for topicName in topics:
                pfeb = self.params_file_extract_bag[topicName]
                rosmsg_class, nb_msgs, dict_corresp, extra = pfeb["msg_class"], pfeb["nb_msgs"], pfeb["columns"], pfeb["extra"]
                # rosmsg_class, dict_corresp, dict_geoindexes=self.params_file_extract_bag[topicName]
                if self.type_extraction == "replace_table":
                    print("Wiping table {}".format(rosmsg_class.__tablename__))
                    self.wipe_table(rosmsg_class)
                countmsg = 0
                print("Uploading {} messages from topic {} in the database table {}".format(
                bag.get_message_count(topic_filters=topicName), topicName, rosmsg_class.__tablename__))
                with tqdm(total=bag.get_message_count(topic_filters=topicName)) as pbar:
                    for subtopic, msg, t in bag.read_messages(topicName):
                        pbar.update(1)
                        countmsg += 1
                        if hasattr(msg, "header"):
                            header_seq = msg.header.seq
                        else :
                            header_seq = countmsg
                        if countmsg == 1:
                            delta_seq = get_delta_seq(self.session, rosmsg_class, header_seq)
                        nseq = header_seq + delta_seq
                        msg_db = rosmsg_class(seq = nseq)
                        msg_db = set_attributes_from_ROSMsg(msg_db, msg, dict_corresp=dict_corresp)
                        msg_db = set_extra(msg_db, extra)

                        self.session.add(msg_db)
                        self.session.commit()

                        if countmsg>=nb_msgs:
                            break

    def extract_csv(self):
        """ Send querries to the database to add the contents of a csv file to the designed db table. """

        pfec = self.params_file_extract_csv
        nb_lines, msg_class, extra = pfec["nb_lines"], pfec["msg_class"], pfec["extra"]
        print("Upload data to database table {} initiated. This may take a while.".format(
        msg_class.__tablename__))
        mat = pd.read_csv(self.file,sep = pfec["sep"], header=pfec["header"],skipinitialspace=True)
        print("File {} opened".format(self.file))
        mat_extracted = DataFrame()
        for i,c in enumerate(pfec["columns"].values()):
            mat_extracted.insert(i, c, mat.loc[:,c])
        if self.type_extraction == "replace_table":
            print("Wiping table {}".format(msg_class.__tablename__))
            self.wipe_table(msg_class)
        if self.session.query(func.count(msg_class.seq)).one()[0] > 0:
            seq = self.session.query(func.max(msg_class.seq)).one()[0]
        else :
            seq = 0
        print("Uploading {} cells in the database table {}".format(
        mat_extracted.shape(), rosmsg_class.__tablename__))
        for ind in mat_extracted.index[0: min(nb_lines, len(mat_extracted.index))]:
            seq += 1
            msg_db = msg_class(seq = seq)
            msg_db = set_attributes_from_csv(msg_db, mat_extracted, pfec["columns"], ind)
            msg_db = set_extra(msg_db, extra)

            self.session.add(msg_db)
            self.session.commit()

    def create_cropped_bag(self, bagfile, pathfile, path_cropped_file, nb_msgs):
        """Create a cropped rosbag out of another one
        The original rosbag is self.bagfile,
        The cropped rosbag file contains the first self.num_msgs_cropped_bag ros
        messages of all messages type.
        """
        time0 = time.time()
        rosbagfile = os.path.join(pathfile, bagfile)
        croppedbagfile = os.path.join(path_cropped_file,"cropped"+str(nb_msgs)+bagfile)
        with rosbag.Bag(croppedbagfile, 'w') as outbag:
            for topic, msg, t in rosbag.Bag(rosbagfile).read_messages():
                if nb_msgs < 1:
                    break
                nb_msgs -= 1
                outbag.write(topic, msg, t)
        timef = time.time()
        string = "Cropped file created in {}s, saved as {}".format(timef-time0, croppedbagfile)
        if self.verbose :
            print(string)
        if self.flg_autolog :
            self.append_logfile(string)

    def create_cropped_csv(self, namefile, pathfile, path_cropped_file, nb_msgs):
        """Create a cropped rosbag out of another one
        The original rosbag is self.bagfile,
        The cropped rosbag file contains the first self.num_msgs_cropped_bag ros
        messages of all messages type.
        """
        time0 = time.time()
        file = os.path.join(pathfile, namefile)
        croppedfile = os.path.join(path_cropped_file,"cropped"+str(nb_msgs)+namefile)
        pfec = self.params_file_extract_csv
        mat = pd.read_csv(file, sep = pfec["sep"], header=pfec["header"], skipinitialspace=True)
        mat_extracted = DataFrame()
        for i,c in enumerate(pfec["columns"].values()):
            mat_extracted.insert(i, c, mat.loc[:nb_msgs,c])
        mat_extracted.to_csv(croppedfile, sep = pfec["sep"], header=pfec["header"])
        timef = time.time()
        string = "Cropped file created in {}s, saved as {}".format(timef-time0, croppedfile)
        if self.verbose :
            print(string)
        if self.flg_autolog :
            self.append_logfile(string)

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
        m_values : ndarray, (1,N)
            The magnetic levels detected at each position of the robot. It is calculated
            as the sum of the influences of all the fake deposits with regard to the
            position of the robot.
        deposits : ndarray, (N,3)
            Positions of the fake deposits generated.
        """
        prd = self.params_random_deposits
        xmin, xmax, ymin, ymax, zmin, zmax = prd["lims"]
        seq, x, y, z, pt = prd["seq"], prd["colx"], prd["coly"], prd["colz"], prd["colpt"]
        flg_proj, avgz, nb_deposits = prd["flg_proj"], prd["avgz"], prd["nb_deposits"]
        mu, sigma, scale = prd["mu"], prd["sigma"], prd["scale"]
        points = extract_points(self.session, seq, x, y, z, pt, flg_proj, [xmin, xmax, ymin, ymax, zmin, zmax])
        # Generation of the fake deposits
        deposits = np.array([[uniform(xmin,xmax), uniform(ymin, ymax), avgz, 1] for i in range(nb_deposits)])
        # Calculation of the distances between each position of the robot and each deposit :
        dists = cdist(points, deposits[:,:3])
        # Addition of a gaussian noise in the background (optionnal)
        # mu, sigma, scale = 1, 0.25, 1 # mean and standard deviation of the
        background = scale*np.random.normal(mu, sigma, len(dists))
        # Calculation of the z values at each position of the robot
        z_values = np.array([np.sum([deposits[i,-1]/(dis[i]**3) + background[i] for i in range(len(deposits))]) for dis in dists])

    def init_table_db(self):
        Base.metadata.create_all(self.engine)

    def wipe_table(self, table):
        t = session.query(table).all()
        for elmt in t :
            session.delete(elmt)

    def append_logfile(self,string):
        return False








# EOF
