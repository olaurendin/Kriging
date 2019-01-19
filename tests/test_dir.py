import rosbag
import os
from pandas import Series, DataFrame
import numpy as np
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

""" Connection parameters to the database """
engine = create_engine('mysql+mysqldb://root:88a002d51c1@localhost/test', echo=True)

""" Creation of a declarative base class """
Base = declarative_base()
# Base.metadata.drop_all(engine)

class Bidon():
    jdfhi= 35
    divfdif = "hdhdhd"

    def __str__(self):
        m = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("_") ]
        m2 = [getattr(self,attr) for attr in m]
        s=""
        for i in range(len(m)):
            s+="{}={}, ".format(m[i], m2[i])
        return "<{}({})>".format(self.__class__.__name__, s[:-2])

class ROSMsg():
    # __tablename__ = 'ROSMsg'
    id = Column(Integer, primary_key=True)

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


class ROSMsg_Float64(Base, ROSMsg):
    __tablename__ = 'float64'

    data = Column(String(40))



if __name__ == "__main__":
    b = Bidon()
    print(b)

    c = ROSMsg()
    print(c)

    d = ROSMsg_Float64()
    print(d)
