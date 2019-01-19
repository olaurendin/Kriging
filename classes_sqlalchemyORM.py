from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Numeric, BigInteger, MetaData, join
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsText, ST_Buffer, ST_Contains
import pyproj


""" Creation of a declarative base class """
Base = declarative_base()

"""
################################################################################
ROSBAG SPECIFIC SQLALCHEMY ORM CLASSES
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

"""
SUBTOPICS ORM CLASSES
"""

class Msg():
    """ Superclass for all msg classes. It defines the common printing methods
    for all msg classes. """

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
    """ Superclass for the header sub-topic of (std_msgs/Header). The seq statement is set as an
    integer and primary_key which is interpreted by sqlalchemy as an auto-increment
    integer key. Thus, this statement will be automatically filled by sqlalchemy core
    if not explictly given. """
    seq = Column(Integer, primary_key=True)     # Autoincrement by default
    time_sec = Column(BigInteger)         # /!\ Not the whole timestamp!
    time_nsec = Column(BigInteger)         # /!\ Not the whole timestamp!
    frame_id = Column(String(20))

class ROSOrientation():
    """ Superclass for the orientation sub-topic (geometry_msgs/Quaternion).
    Does not provide a primary key."""
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

# class ROSMsg_boat_heading(Base, Msg):
#     __tablename__ = 'boat_heading'
#     id = Column(Integer, primary_key=True)
#     data = Column(Numeric(precision = 6, scale = 4))
#
# class ROSMsg_imu(Base, Msg, ROSHeader, ROSOrientation, ROSAngular_velocity, ROSLinear_acceleration):
#     __tablename__ = 'imu'
#
# class ROSMsg_gps_course(Base, Msg):
#     __tablename__ = 'gps_course'
#     id = Column(Integer, primary_key=True)
#     data = Column(Numeric(precision = 6, scale = 4))

class ROSMsg_fix(Msg, ROSHeader):
    lat = Column(Numeric(precision = 10, scale = 8))
    lon = Column(Numeric(precision = 10, scale = 8))
    alt = Column(Numeric(precision = 6, scale = 3))
    pt = Column(Geometry('POINT'))      # The geo index is created at the same time by default

class ROSMsg_fix_107(Base, ROSMsg_fix):
    __tablename__ = 'fix_107'

class ROSMsg_fix_117(Base, ROSMsg_fix):
    __tablename__ = 'fix_117'

class ROSMsg_fix_137(Base, ROSMsg_fix):
    __tablename__ = 'fix_137'

class ROSMsg_fix_117_test(Base, ROSMsg_fix):
    __tablename__ = 'fix_117_test'

"""
################################################################################
OTHER SQLALCHEMY ORM CLASSES
"""

# class ROSMsg_Base(Base, Msg):
#     __tablename__ = 'test2'
#     seq = Column(Integer, primary_key=True)     # Autoincrement by default

class ROSMsg_fix_test(Base, ROSMsg_fix):
    __tablename__ = 'fix_test'

class ROSMsg_test2(Base):
    __tablename__ = 'test2'
    seq = Column(Integer, primary_key=True)     # Autoincrement by default
    data = Column(Numeric(precision = 8, scale = 4))

class Msg_nav_data(Base, Msg):
    __tablename__ = 'nav_data'
    seq = Column(Integer, primary_key=True)     # Autoincrement by default
    time = Column(Numeric(precision = 6, scale = 2))
    x = Column(Numeric(precision = 8, scale = 4))
    y = Column(Numeric(precision = 8, scale = 4))
    depth = Column(Numeric(precision = 8, scale = 4))
    a = Column(Numeric(precision = 8, scale = 4))
    depth_seabed = Column(Numeric(precision = 8, scale = 4))
    pt = Column(Geometry('POINT'))      # The geo index is created at the same time by default


# class Base_table(Base, Msg):
#     seq = Column(Integer, primary_key=True)     # Autoincrement by default
#
# class Random_deposits_base(Base_table):
#     __tablename__ = "random_deposits_table"
#     x = Column(Numeric(precision = 8, scale = 4))
#     y = Column(Numeric(precision = 8, scale = 4))
#     z = Column(Numeric(precision = 8, scale = 4))
#
# class Random_Z_values(Base_table):
#     __tablename__ = "random_Z_values"
#     Z = Column(Numeric(precision = 8, scale = 4))


#EOF
