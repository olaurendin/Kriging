
"""
SOURCE : SQLAlchemy Expression Language Tutorial dedicated to the core of the
library, in charge of the translation of basic SQL statements to the DBAPI at a
low level of abstraction.
https://docs.sqlalchemy.org/en/latest/core/tutorial.html
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

""" Table creation """
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
metadata = MetaData()       # Gathers all informations of the database
# The table itself :
users = Table('users', metadata,
     Column('id', Integer, primary_key=True),
     Column('name', String(10)),
     Column('fullname', String(50)),
     # mysql_engine='MyISAM'
)

addresses = Table('addresses', metadata,
   Column('id', Integer, primary_key=True),
   Column('user_id', None, ForeignKey('users.id')),
   Column('email_address', String(40), nullable=False)
  )

# users.drop(engine, checkfirst=True) # Drop the former table "users" if exists
# users.create(engine, checkfirst=True) # Create the table "users" if doesn't exist already
metadata.drop_all(engine)         # Drop all tables
metadata.create_all(engine)         # Create all new tables

""" INSERT statement """
ins = users.insert().values(name='jack', fullname='Jack Jones')     # Insert values
print(str(ins))                  # As we can see, each value is linked to a bind parameter)
print(ins.compile().params)     # These parameters are stocked in the instance ins, we can plot
                                # the values of the parameters that way


""" Connection to the database and commit of the previous statements """
conn = engine.connect()
result = conn.execute(ins)      # "Cursor" of the DBAPI linked to the INSERT statement
                                # Can get valuable info out of it
print(result.inserted_primary_key)

""" Executing multiple statements """
ins = users.insert()
conn.execute(ins, id=2, name='wendy', fullname='Wendy Williams')    # One single statement
conn.execute(addresses.insert(), [
    {'user_id': 1, 'email_address' : 'jack@yahoo.com'},
    {'user_id': 1, 'email_address' : 'jack@msn.com'},
    {'user_id': 2, 'email_address' : 'www@www.org'},
    {'user_id': 2, 'email_address' : 'wendy@aol.com'},
])

""" SELECT statement """
from sqlalchemy.sql import select
s = select([users])             # "SELECT * FROM users" by default
# s = select([users.c.name, users.c.fullname])

# Reading the selected rows
# Considering the result of type ResultProxy as an iterable :
result = conn.execute(s)
for row in result:
     print(row)
result.close()          # Not compusolry
# Or accessing the rows by their names or indexes
result = conn.execute(s)
row = result.fetchone()
print("name:", row['name'], "; fullname:", row['fullname'])
print("name:", row[1], "; fullname:", row[2])
# Or through the Column objects :
print("name:", row[users.c.name], "; fullname:", row[users.c.fullname])
result.close()

""" cartesian product between two tables """
# for row in conn.execute(select([users, addresses])):
#     print(row)

""" Join tables with a WHERE statement """
# s = select([users, addresses]).where(users.c.id == addresses.c.user_id)
# for row in conn.execute(s):
#     print(row)

""" Conjunctions """
# Here we use the and_ and or_ operators to select all users who have an email
# address at AOL or MSN, whose name starts with a letter between “m” and “z”,
# and we’ll also generate a column containing their full name combined with
# their email address.
# It is also possible to replace the and_ keywords with a chain of .where()
from sqlalchemy.sql import and_, or_, not_
s = select([(users.c.fullname +
              ", " + addresses.c.email_address).
               label('title')]).\
       where(
          and_(
              users.c.id == addresses.c.user_id,
              users.c.name.between('m', 'z'),
              or_(
                 addresses.c.email_address.like('%@aol.com'),
                 addresses.c.email_address.like('%@msn.com')
              )
          )
       )
result = conn.execute(s)
print(result.fetchall())
result.close()

"""Using textual SQL"""
from sqlalchemy.sql import text
s = text(
# For other databases drivers than mySQL :
# "SELECT users.fullname || ', ' || addresses.email_address AS title "
    "SELECT CONCAT(users.fullname , ', ' , addresses.email_address) AS title "
        "FROM users, addresses "
        "WHERE users.id = addresses.user_id "
        "AND users.name BETWEEN :x AND :y "
        "AND (addresses.email_address LIKE :e1 "
            "OR addresses.email_address LIKE :e2)")
result = conn.execute(s, x='m', y='z', e1='%@aol.com', e2='%@msn.com')
print(result.fetchall())
result.close()

# Note : It could be usefull to specify Bound Parameters :
stmt = text("SELECT * FROM users WHERE users.name BETWEEN :x AND :y")
stmt = stmt.bindparams(x="m", y="z")
# Or explicitly :
# stmt = stmt.bindparams(bindparam("x", type_=String), bindparam("y", type_=String))
result = conn.execute(stmt, {"x": "m", "y": "z"})
result.close()



s=text("SET FOREIGN_KEY_CHECKS = 0;"
"DROP TABLE IF EXISTS addresses;"
"DROP TABLE IF EXISTS users;"
"SET FOREIGN_KEY_CHECKS = 1;")
conn.execute(s)


# engine.dispose()
