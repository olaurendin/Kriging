
from os import walk, listdir, system
from rosbag2postgresql import rosbag_to_postgresql
import rosbag
import os
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
import tracemalloc
from concurrent.futures import ProcessPoolExecutor

# def multiprocessing(func, args, workers):
#     with ProcessPoolExecutor(workers) as ex:
#         res = ex.map(func, args)
#     print(list(res))
#     return list(res)
#
# def pickling_func(rtp, session, topicstoext, delta_seq):
#     return rtp.update_database(session, topicstoext, delta_seq)

if __name__ == "__main__":

    # tracemalloc.start(10)

    """ Connection parameters to the database """
    engine = create_engine('postgresql+psycopg2://ganlan:88a002d51c1@localhost/dataAUV', echo=False)

    path = "/home/ganlan/internship/Final_code/data/Boatbot-exp/137/boatbot_zotac137"
    folders = listdir(path)
    folders.sort()
    delta_seq = 0
    for folder in folders :
        folder = os.path.join(path, folder)
        print(folder)

        """ Creation of a rosbag to postgresql instance """
        rtp = rosbag_to_postgresql(pathfile=folder, engine=engine, init_db=True)

        """ Creation of all the tables (If it wasn't created at the init with init_db = True) """
        # rtp.init_database(Base)

        """ Connection to the database """
        Session = sessionmaker(bind=engine)
        session = Session()

        """ Extracting messages from rosbag to database """
        topicstoext = ["/zodiac_auto/fix"]
        # multiprocessing(func = pickling_func, args = (rtp, session, topicstoext, delta_seq), \
        #                 workers = 1)
        delta_seq = rtp.update_database(session, topicstoext, delta_seq)

        print("\n\n################# NEW DELTA_SEQ : "+str(delta_seq)+"\n\n")

        """ Sending querries to the database """
        # msg_boat_heading = ROSMsg_boat_heading(data=644269)
        # session.add(ed_user)
        # session.add_all([
        #     ROSMsg_fix(time_nsec = 1, frame_id = "a", lat = 2, lon = 3, alt = 4, pocov = 5, pt='POINT(0 0)'),
        #     ROSMsg_fix(time_nsec = 1, frame_id = "a", lat = 2, lon = 3, alt = 4, pocov = 5, pt='POINT(0 1)'),
        # ])
        # print(session.new)

        # Commit all addtions/modifications
        # session.commit()

        """ Print all points in the neighborhood of a given point """
        # point = session.query(ST_AsText(ROSMsg_fix.pt)).order_by(ROSMsg_fix.seq).first()
        # allpoints = session.query(ST_AsText(ROSMsg_fix.pt)).order_by(ROSMsg_fix.seq).all()
        # print(point)
        # circle = ST_Buffer(point,2)
        # print(session.query(ST_AsText(circle)).all())
        # sub = session.query(ST_AsText(ROSMsg_fix.pt)).\
        #         filter(ST_Contains(circle,ROSMsg_fix.pt)).\
        #         all()
        # print(sub)
        # session.commit()

        # allpoints = session.query(ST_AsText(ROSMsg_fix.pt)).order_by(ROSMsg_fix.seq).all()
        # alts = session.query(ROSMsg_fix.alt).order_by(ROSMsg_fix.seq).all()
        # # print(allpoints)
        # session.commit()
        # nb_msgs = len(allpoints)
        # print(nb_msgs)
        # print(allpoints[0][0], allpoints[1][0], allpoints[0][0]+allpoints[1][0])
        # print(alts[0][0], alts[1][0], alts[0][0]+alts[1][0])
        # points = np.zeros([nb_msgs, 3])
        # for i in range(nb_msgs):
        #     points[i,:] = float(allpoints[i][0][6:-1].split(" ")[0]), float(allpoints[i][0][6:-1].split(" ")[1]), alts[i][0]
        # print(points)
        #
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot(points[:,0], points[:,1], points[:,2])
        # ax.legend()
        #
        # plt.show()

























    print("\nEOF")
