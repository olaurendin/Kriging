from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String


""" Connection parameters to the database """
engine = create_engine('mysql+mysqldb://root:88a002d51c1@localhost/test', echo=True)

""" Creation of a declarative base class """
Base = declarative_base()
Base.metadata.drop_all(engine)

""" Definition of a python class into the Base """
class Sdata(Base):
    __tablename__ = 'sdata'

    t = Column(Integer, primary_key=True)
    boat_heading = Column(String(40))
    fix = Column(String(40))               
    password = Column(String(40))

    #  Optionnal, just for plotting
    def __repr__(self):
       return "<User(name='%s', fullname='%s', password='%s')>" % (
                            self.name, self.fullname, self.password)

    def __str__(self):
       return "<User(name='%s', fullname='%s', password='%s')>" % (
                            self.name, self.fullname, self.password)
