"""
Adapted from the rosbag to csv python 2 code written by Nick Speal in May 2013
at McGill University's Aerospace Mechatronics Laboratory
www.speal.ca

Supervised by Professor Inna Sharf, Professor Meyer Nahon
http://www.clearpathrobotics.com/assets/guides/ros/Converting%20ROS%20bag%20to%20CSV.html

Adapted for python3, tested on python3.5
"""


""" MEMO ROSMSGS """
# /zodiac_auto/boat_heading       std_msgs/Float64
    # float64 data
# /zodiac_auto/imu                sensor_msgs/Imu
    # std_msgs/Header header
    #   uint32 seq
    #   time stamp
    #   string frame_id
    # geometry_msgs/Quaternion orientation
    #   float64 x
    #   float64 y
    #   float64 z
    #   float64 w
    # float64[9] orientation_covariance
    # geometry_msgs/Vector3 angular_velocity
    #   float64 x
    #   float64 y
    #   float64 z
    # float64[9] angular_velocity_covariance
    # geometry_msgs/Vector3 linear_acceleration
    #   float64 x
    #   float64 y
    #   float64 z
    # float64[9] linear_acceleration_covariance
# /zodiac_auto/gps_course         std_msgs/Float64
    # float64 data
# /zodiac_auto/time_reference     sensor_msgs/TimeReference
    # std_msgs/Header header
    #   uint32 seq
    #   time stamp
    #   string frame_id
    # time time_ref
    # string source
# /zodiac_auto/fix                sensor_msgs/NavSatFix
    # uint8 COVARIANCE_TYPE_UNKNOWN=0
    # uint8 COVARIANCE_TYPE_APPROXIMATED=1
    # uint8 COVARIANCE_TYPE_DIAGONAL_KNOWN=2
    # uint8 COVARIANCE_TYPE_KNOWN=3
    # std_msgs/Header header
    #   uint32 seq
    #   time stamp
    #   string frame_id
    # sensor_msgs/NavSatStatus status
    #   int8 STATUS_NO_FIX=-1
    #   int8 STATUS_FIX=0
    #   int8 STATUS_SBAS_FIX=1
    #   int8 STATUS_GBAS_FIX=2
    #   uint16 SERVICE_GPS=1
    #   uint16 SERVICE_GLONASS=2
    #   uint16 SERVICE_COMPASS=4
    #   uint16 SERVICE_GALILEO=8
    #   int8 status
    #   uint16 service
    # float64 latitude
    # float64 longitude
    # float64 altitude
    # float64[9] position_covariance
    # uint8 position_covariance_type

"""##########################################################################"""

import rosbag
import os
from os import walk, listdir, system
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, BigInteger, MetaData, join
# from sqlalchemy.types import DECIMAL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsText, ST_Buffer, ST_Contains
import pyproj
import gc
import tracemalloc
import sys
from concurrent.futures import ProcessPoolExecutor

# gc.set_threshold(0)

def multiprocessing(func, args, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    print(list(res))
    return list(res)

# def pickling_func(rtp, session, topicstoext, delta_seq):
#     return rtp.update_database(session, topicstoext, delta_seq)

def pickling_func2(folder, engine, delta_seq):
        # tracemalloc.start(10)

        """ Connection parameters to the database """
        engine = create_engine('postgresql+psycopg2://ganlan:88a002d51c1@localhost/dataAUV_117', echo=False)

        """ Creation of a rosbag to postgresql instance """
        rtp = rosbag_to_postgresql(pathfile=folder, engine=engine, init_db=True)

        """ Creation of all the tables (If it wasn't created at the init with init_db = True) """
        # rtp.init_database(Base)

        """ Connection to the database """
        Session = sessionmaker(bind=engine)
        session = Session()

        """ Extracting messages from rosbag to database """
        topicstoext = ["/zodiac_auto/fix"]
        return rtp.update_database(session, topicstoext, delta_seq)

        # delta_seq = rtp.update_database(session, topicstoext, delta_seq)

        # print("\n\n################# NEW DELTA_SEQ : "+str(delta_seq)+"\n\n")


class rosbag_to_postgresql():

    def __init__(self, bagfile = "", pathfile = ".", engine=None, num_msgs = 10000, \
            cropped_flg = False, init_db = False, verbose = False):
        self.bagfile = bagfile
        self.pathfile = pathfile
        self.num_msgs = num_msgs
        self.cropped_flg = cropped_flg
        self.verbose = verbose
        self.init_db = init_db
        self.snapshots = []
        if engine != None:
            self.engine = engine
        else :
            print("No engine selected, connection to the database aborted")
            os.abort()
        if self.cropped_flg:
            self.create_cropped_rosbag()
        if bagfile != "":
            self.rosbagfiles = [os.path.join(self.pathfile, self.bagfile)]
        else :
            self.rosbagfiles = [os.path.join(self.pathfile,f) for f in os.listdir(self.pathfile) if f[-4:] == ".bag"]
        if self.init_db:
            self.init_database(Base)


    def create_cropped_rosbag(self):
        time0 = time.time()
        nb_msgs = self.num_msgs
        self.rosbagfiles = os.path.join(self.pathfile, self.bagfile)
        croppedbagfile = os.path.join(self.pathfile,"cropped"+str(self.num_msgs)+self.bagfile)
        with rosbag.Bag(croppedbagfile, 'w') as outbag:
            for topic, msg, t in rosbag.Bag(self.rosbagfiles).read_messages():
                if nb_msgs < 1:
                    break
                nb_msgs -= 1
                outbag.write(topic, msg, t)
        self.bagfile = croppedbagfile
        timef = time.time()
        print("cropped file finished in {}s".format(timef-time0))

    def update_database(self, session, topicstoext, delta_seq = 0):
        inProj = pyproj.Proj(init='epsg:4326')      # Latlong WSG84
        outProj = pyproj.Proj(init='epsg:2154')     # Lambert-93

        numberOfFiles = str(len(self.rosbagfiles))
        print ("reading all {} bagfiles in current directory: \n".format(numberOfFiles))
        for f in self.rosbagfiles:
            print (f)

        count = 0
        timeinit = time.time()
        for bagFile in self.rosbagfiles:

            print("_____________________________________________________________\n")

            time0 = time.time()
            count += 1
            print ("reading file {} of {} : {}\n".format(count, numberOfFiles, bagFile))
            #access bag
            with rosbag.Bag(bagFile, 'r') as bag:
                # bag = rosbag.Bag(bagFile)
                # bagContents = bag.read_messages()
                bagName = bag.filename
                time1 = time.time()
                print("Finished importing file {} in {}s\n".format(bagName, time1-time0))

                topics = list(bag.get_type_and_topic_info()[1].keys())
                msgs = list(bag.get_type_and_topic_info()[1].values())
                self.topics = topics
                self.msgs = msgs
                types = [msgs[i].msg_type for i in range(len(msgs))]
                if self.verbose :
                    print("\n\ntopics :", topics)
                    print("\n\nmsgs :", msgs)
                    print ("\n\ntypes :", types)
                self.nbmessages = [msgs[i].message_count for i in range(len(msgs))]
                self.nbmessages_tot = sum(self.nbmessages)

                time2 = time.time()
                print("list of topics : {} found in {}s\n".format(topics, time2-time1))

                topics = [topic for topic in topics if topic in topicstoext]
                nmsgmax = 0
                """for topicName in topics:
                    print(topicName)
                    countmsg = 0
                    for subtopic, msg, t in bag.read_messages(topicName):
                        countmsg += 1
                        # print(countmsg)
                        x,y = pyproj.transform(inProj,outProj,msg.longitude,msg.latitude)
                        # print(msg.latitude,msg.longitude,x,y)
                        nseq = msg.header.seq + delta_seq
                        msg = ROSMsg_fix(seq = nseq,
                                    time_sec = msg.header.stamp.secs,
                                    time_nsec = msg.header.stamp.nsecs,
                                    frame_id = msg.header.frame_id,
                                    lat = msg.latitude,
                                    lon = msg.longitude,
                                    alt = msg.altitude,
                                    # pocov = msg.position_covariance,
                                    pt='POINT({} {})'.format(x,y))
                        session.add_all([msg])
                        del(msg)
                        if nseq > nmsgmax :
                            nmsgmax = nseq
                        if countmsg>=1000:
                            session.commit()
                            print(countmsg)
                            # self.collect_stats()
                            countmsg = 0
                session.commit()"""
                # bag.close()
                time3 = time.time()
                # del(self.nbmessages_tot)
                # del(self.nbmessages)
                # del(topics)
                # del(msgs)
                # del(types)
                print("Finished updating database for file {} in {}s\n".format(bagName, time3-time2))
                # del(bagName)
                # del(bagContents)



                # bag = None
                # del(bag)
                print("refcount 1 : ", sys.getrefcount(bag))
                print(len(gc.get_referrers(bag)))
                print(gc.get_referrers(bag))
                print("\n\n")

                print("refcount 2 : ", sys.getrefcount(rosbag.Bag(bagFile, 'r')))
                print(len(gc.get_referrers(rosbag.Bag(bagFile, 'r'))))
                print(gc.get_referrers(rosbag.Bag(bagFile, 'r')))
                print("\n\n")

                # del(rosbag.Bag(bagFile, 'r'))
                del(bag)

                print("refcount 3 : ", sys.getrefcount(rosbag.Bag(bagFile, 'r')))
                print(len(gc.get_referrers(rosbag.Bag(bagFile, 'r'))))
                print(gc.get_referrers(rosbag.Bag(bagFile, 'r')))


                gc.collect()
            # print("refcount 1 : ", sys.getrefcount(bag))

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        timeend = time.time()
        print("Finished updating database for all files in {}s\n".format(timeend-timeinit))
        return delta_seq+nmsgmax

    def init_database(self, Base):
        Base.metadata.create_all(self.engine)

    def collect_stats(self):
        self.snapshots.append(tracemalloc.take_snapshot())
        if len(self.snapshots) > 1:
            stats = self.snapshots[-1].filter_traces(filters).compare_to(self.snapshots[-2], 'filename')

            for stat in stats[:10]:
                print("{} new KiB {} total KiB {} new {} total memory blocks: ".format(stat.size_diff/1024, stat.size / 1024, stat.count_diff ,stat.count))
                for line in stat.traceback.format():
                    print(line)

""" Creation of a declarative base class """
Base = declarative_base()

"""
SUBTOPICS OSM CLASSES
"""

class ROSMsg():

    def __repr__(self):
        m = [attr for attr in dir(self) if not callable(getattr(self, attr))
                and not attr.startswith("_")]
        m2 = [getattr(self,attr) for attr in m]
        s=""
        for i in range(len(m)):
            s+="{}={}, ".format(m[i], m2[i])
        return "<{}({})>".format(self.__class__.__name__, s[:-2])

    def __str__(self):
        m = [attr for attr in dir(self) if not callable(getattr(self, attr))
                and not attr.startswith("_")]
        m2 = [getattr(self,attr) for attr in m]
        s=""
        for i in range(len(m)):
            s+="{}={}, ".format(m[i], m2[i])
        return "<{}({})>".format(self.__class__.__name__, s[:-2])

class ROSHeader():
    seq = Column(Integer, primary_key=True)
    time_sec = Column(BigInteger)         # /!\ Not the whole timestamp!
    time_nsec = Column(BigInteger)         # /!\ Not the whole timestamp!
    frame_id = Column(String(20))

class ROSOrientation():
    qx = Column(Numeric(precision = 6, scale = 4))
    qy = Column(Numeric(precision = 6, scale = 4))
    qz = Column(Numeric(precision = 6, scale = 4))
    qw = Column(Numeric(precision = 6, scale = 4))
    ocov = Column(Numeric(precision = 6, scale = 4))

class ROSAngular_velocity():
    avx = Column(Numeric(precision = 6, scale = 4))
    avy = Column(Numeric(precision = 6, scale = 4))
    avz = Column(Numeric(precision = 6, scale = 4))
    avcov = Column(Numeric(precision = 6, scale = 4))

class ROSLinear_acceleration():
    lax = Column(Numeric(precision = 6, scale = 4))
    lay = Column(Numeric(precision = 6, scale = 4))
    laz = Column(Numeric(precision = 6, scale = 4))

"""
TOPICS OSM CLASSES : WILL CREATE A CORRESPONDING TABLE
"""

# class ROSMsg_boat_heading(Base, ROSMsg):
#     __tablename__ = 'boat_heading'
#     id = Column(Integer, primary_key=True)
#     data = Column(Numeric(precision = 6, scale = 4))
#
# class ROSMsg_imu(Base, ROSMsg, ROSHeader, ROSOrientation, ROSAngular_velocity, ROSLinear_acceleration):
#     __tablename__ = 'imu'
#
# class ROSMsg_gps_course(Base, ROSMsg):
#     __tablename__ = 'gps_course'
#     id = Column(Integer, primary_key=True)
#     data = Column(Numeric(precision = 6, scale = 4))

class ROSMsg_fix(Base, ROSMsg, ROSHeader):
    __tablename__ = 'fix'
    lat = Column(Numeric(precision = 6, scale = 4))
    lon = Column(Numeric(precision = 6, scale = 4))
    alt = Column(Numeric(precision = 6, scale = 4))
    # pocov = Column(Numeric(precision = 6, scale = 4))
    # pt = Column(Point(precision = 6, scale = 4, nullable = False))
    pt = Column(Geometry('POINT'))      # The geo index is created at the same time by default
































# if __name__ == "__main__":
#     bagfile = "rosbag_2018-07-11-10-23-16_0.bag"
#     pathfile = "/home/ganlan/internship/Final_code/data/Boatbot-exp/117/boatbot_zotac117/2018-07-11-10:23:05"
#     inputfile = os.path.join(pathfile, bagfile)
#     num_msgs = 10
#     outputfile = os.path.join(pathfile, "cropped"+str(num_msgs)+bagfile)
#
#     """ Connection parameters to the database """
#     engine = create_engine('postgresql+psycopg2://ganlan:88a002d51c1@localhost/opengeo', echo=False)
#
#     """ Creation of a rosbag to postgresql instance """
#     rtp = rosbag_to_postgresql(bagfile=bagfile, pathfile=pathfile, engine=engine, num_msgs=num_msgs, cropped_flg=True)
#
#     """ Creation of all the tables (If they weren't created at the init with init_db = True)"""
#     # rtp.init_database(Base)
#
#     """ Connection to the database """
#     Session = sessionmaker(bind=engine)
#     session = Session()
#
#     """ Extracting messages from rosbag to database """
#     topicstoext = ["/zodiac_auto/fix"]
#     rtp.update_database(session, topicstoext)
#
#     """ Sending querries to the database """
#     # msg_boat_heading = ROSMsg_boat_heading(data=644269)
#     # session.add(ed_user)
#     # session.add_all([
#     #     ROSMsg_fix(time_nsec = 1, frame_id = "a", lat = 2, lon = 3, alt = 4, pocov = 5, pt='POINT(0 0)'),
#     #     ROSMsg_fix(time_nsec = 1, frame_id = "a", lat = 2, lon = 3, alt = 4, pocov = 5, pt='POINT(0 1)'),
#     # ])
#     # print(session.new)
#
#     # Commit all addtions/modifications
#     # session.commit()
#
#     """ Print all points in the neighborhood of a given point """
#     # point = session.query(ST_AsText(ROSMsg_fix.pt)).order_by(ROSMsg_fix.seq).first()
#     # allpoints = session.query(ST_AsText(ROSMsg_fix.pt)).order_by(ROSMsg_fix.seq).all()
#     # print(point)
#     # circle = ST_Buffer(point,2)
#     # print(session.query(ST_AsText(circle)).all())
#     # sub = session.query(ST_AsText(ROSMsg_fix.pt)).\
#     #         filter(ST_Contains(circle,ROSMsg_fix.pt)).\
#     #         all()
#     # print(sub)
#     # session.commit()
#
#     allpoints = session.query(ST_AsText(ROSMsg_fix.pt)).order_by(ROSMsg_fix.seq).all()
#     alts = session.query(ROSMsg_fix.alt).order_by(ROSMsg_fix.seq).all()
#     # print(allpoints)
#     session.commit()
#     nb_msgs = len(allpoints)
#     print(nb_msgs)
#     print(allpoints[0][0], allpoints[1][0], allpoints[0][0]+allpoints[1][0])
#     print(alts[0][0], alts[1][0], alts[0][0]+alts[1][0])
#     points = np.zeros([nb_msgs, 3])
#     for i in range(nb_msgs):
#         points[i,:] = float(allpoints[i][0][6:-1].split(" ")[0]), float(allpoints[i][0][6:-1].split(" ")[1]), alts[i][0]
#     print(points)
#
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot(points[:,0], points[:,1], points[:,2])
#     ax.legend()
#
#     plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     print("\nEOF")

if __name__ == "__main__":

    # tracemalloc.start(10)

    """ Connection parameters to the database """
    engine = create_engine('postgresql+psycopg2://ganlan:88a002d51c1@localhost/opengeo', echo=False)

    path = "/home/ganlan/internship/Final_code/data/Boatbot-exp/117/boatbot_zotac117"
    folders = listdir(path)
    delta_seq = 0
    for folder in folders :
        folder = os.path.join(path, folder)
        print(folder)
        delta_seq=multiprocessing(func = pickling_func2, args = (folder, engine, delta_seq), \
                        workers = 1)
#
#         """ Creation of a rosbag to postgresql instance """
#         rtp = rosbag_to_postgresql(pathfile=folder, engine=engine, init_db=True)
#
#         """ Creation of all the tables (If it wasn't created at the init with init_db = True) """
#         # rtp.init_database(Base)
#
#         """ Connection to the database """
#         Session = sessionmaker(bind=engine)
#         session = Session()
#
#         """ Extracting messages from rosbag to database """
#         topicstoext = ["/zodiac_auto/fix"]
#         multiprocessing(func = pickling_func, args = (rtp, session, topicstoext, delta_seq), \
#                         workers = 1)
#         # delta_seq = rtp.update_database(session, topicstoext, delta_seq)
#
#         print("\n\n################# NEW DELTA_SEQ : "+str(delta_seq)+"\n\n")
