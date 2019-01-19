"""
SOURCE : SQLAlchemy Expression Language Tutorial dedicated to the OSM (Object
Relationnal Mapper) which does the correspondance between the data stored in the
database and the corresponding python objects>
https://docs.sqlalchemy.org/en/latest/orm/tutorial.html
"""

import sqlalchemy

""" Connection parameters to the database """
# No connection with the database have been established at this point,
# SQLAlchemy's core just makes sure the corresponding database exist
# create_engine("dialect+driver://username:password@host:port/database")
# Engine configuration documetation :
# https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
from sqlalchemy import create_engine
engine = create_engine('mysql+mysqldb://root:88a002d51c1@localhost/test', echo=True)

""" Creation of a declarative base class """
# Sums up the database tables we’ll be dealing with, and our corresponding classes
# https://docs.sqlalchemy.org/en/latest/orm/extensions/declarative/index.html
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
Base.metadata.drop_all(engine)

""" Definition of a python class into the Base """
from sqlalchemy import Column, Integer, String
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)      # Note that the class in the Base
    name = Column(String(40))                       # Only possesses attributes part
    fullname = Column(String(40))                   # of the sqlalchemy standard types
    password = Column(String(40))                   # And these are CLASSES ATTRIBUTES

    #  Optionnal, just for plotting
    def __repr__(self):
       return "<User(name='%s', fullname='%s', password='%s')>" % (
                            self.name, self.fullname, self.password)

    def __str__(self):
       return "<User(name='%s', fullname='%s', password='%s')>" % (
                            self.name, self.fullname, self.password)
print(str(User.__table__))      # the corresponding table is actually created at the same tine

""" Creation of all the tables """
Base.metadata.create_all(engine)

""" Adding an entry """
ed_user = User(name='ed', fullname='Ed Jones', password='edspassword')
print(ed_user)
print(ed_user.name)
print(ed_user.id)

""" Creating a session : communication with the database """
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
session = Session()
# See also Session = sessionmaker() and Session.configure(bind=engine)

# Addind and updating objects :
ed_user = User(name='ed', fullname='Ed Jones', password='edspassword')
session.add(ed_user)    # user still pending, will be committed to the db if needed
print(session.new)             # Lists all pending rows
our_user = session.query(User).filter_by(name='ed').first() # query a user info
# We have asked for the info regarding the "ed" user, the former row has been
# commited to the db so that the query can occur
print(our_user)     # turns out the python object returned is the same than ed

# Modifying object :
ed_user.password = 'f8s7ccs'
print(session.dirty)    # The "ed" user information has been modified, still pending

# Adding multiple objects :
session.add_all([
    User(name='wendy', fullname='Wendy Williams', password='foobar'),
    User(name='mary', fullname='Mary Contrary', password='xxg527'),
    User(name='fred', fullname='Fred Flinstone', password='blah')])

# Commit all addtions/modifications
session.commit()

print(ed_user.id)   # Note that now the autoincrement key id has been fulfilled

""" Rollback """
# Get back to the previous state of the db in case of an erroneous operation
ed_user.name = 'Edwardo'    # erroneous modifications
fake_user = User(name='fakeuser', fullname='Invalid', password='12345')
session.add(fake_user)      # Erroneous addition
session.query(User).filter(User.name.in_(['Edwardo', 'fakeuser'])).all()    # Unwanted result
session.rollback()          # Rollback to the previous state of the db
session.query(User).filter(User.name.in_(['ed', 'fakeuser'])).all()     # previous state of the db





# Base.metadata.drop_all(engine)
