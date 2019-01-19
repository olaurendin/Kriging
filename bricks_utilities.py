import numpy as np
import time
from sqlalchemy import create_engine, between
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from geoalchemy2.functions import ST_AsText, ST_Buffer, ST_Contains
import matplotlib.pyplot as plt
import pyproj

from proj_utilities import proj_xy

### Bricks : three dimensional parallelogram where points will be extracted from.
### Each item of the dictionnary corresponds to the number of the brick selected.
### For now, this bricks are only taken from the files of 13/07/18 (ROSMsg_fix_137 database)
### Pattern : { "name_brick" : (xmin,xmax,ymin,ymax,altmin,altmax)}
### The value "False" means that the border value of the data positions will be selected.
bricks = { "default" : (False, False, False, False, False, False),
"4+5+6+11" : (132000, 136000, 6831950, 6833000, False, False),
"11" : (132750, 134250, 6831900, 6832200, 0, False),
"4" : (132800, 134200, 6831200, 6831470, 53.7, False),
"5+6" : (132890, 134227, 6831340, 6831800, 50, 52.5)
}

def create_rect(xmin, xmax, ymin, ymax):
    """
    Create a POLYGON instance in the shape of a rectangle
    """
    rect = 'POLYGON(({} {},{} {},{} {},{} {}, {} {}))'.format(xmin,ymin, xmin,ymax,\
                                        xmax,ymax, xmax,ymin, xmin,ymin)
    return rect

def extract_points_old(rosmsg, session, xmin=False, xmax=False, ymin=False, ymax=False,
                altmin=False, altmax=False):
    """Extract the positions of the robot in a brick given its spatial delimitation

    Parameters
    ----------
    rosmsg : ROSMsg subclass instance
        The SQLAlcheny ORM Base instance to communicate with the corresponding table
        in the Postgresql database.
    session : sqlalchemy.orm.session.Session
        SQLAlchemy Session instance to send queries with.
    xmin, xmax, ymin, ymax, altmin, altmax : float
        Spatial boundaries of the brick to extract.

    Returns
    -------
    points : ndarray, (N,3)
        Positions of the robot in the given brick.
    """
    xtotmin, xtotmax, ytotmin, ytotmax = session.query(\
    func.min(rosmsg.lon),func.max(rosmsg.lon),\
    func.min(rosmsg.lat),func.max(rosmsg.lat)).one()
    alttotmin, alttotmax = session.query(\
    func.min(rosmsg.alt), func.max(rosmsg.alt)).one()
    x1,y1 = pyproj.transform(rosmsg.inProj(),rosmsg.outProj(),xtotmin,ytotmin)
    x2,y2 = pyproj.transform(rosmsg.inProj(),rosmsg.outProj(),xtotmax,ytotmax)
    # l1 = [xmin,xmax,ymin,ymax]
    # l2 = [altmin, altmax]
    # l3 = [x1,x2,y1,y2]
    # l4 = [alttotmin, alttotmax]
    # l1 = [l3[i] if not l1[i] else l1[i] for i in range(len(l1))]
    # l2 = [l4[i] if not l2[i] else l2[i] for i in range(len(l2))]
    l1 = smallest_list([xmin,xmax,ymin,ymax], [x1,x2,y1,y2])
    l2 = smallest_list([altmin, altmax], [alttotmin, alttotmax])
    print(l1, l2)
    xmin,xmax,ymin,ymax = l1
    altmin, altmax = l2
    rect = create_rect(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    sub = session.query(ST_AsText(rosmsg.pt), rosmsg.alt).\
            filter(ST_Contains(rect,rosmsg.pt)).\
            filter(rosmsg.alt>altmin).\
            filter(rosmsg.alt<altmax).\
            all()
    nb_msgs = len(sub)
    points = np.zeros([nb_msgs, 3])
    for i in range(nb_msgs):
        points[i,:] = float(sub[i][0][6:-1].split(" ")[0]), float(sub[i][0][6:-1].split(" ")[1]), sub[i][1]
    return points

def smallest_list(l1, l2):
    """
    Get the smallest list (understand the the one representing the smallest interval
    of values) composed of the elements of l1 and l2.

    Parameters
    ----------
    l1 : List of ints, floats or False values of length n
        If False is given at a given index i, this element will be replaced  in l by
        the ith element of list l2 of length n
    l2 : List of ints or floats

    Returns
    -------
    l : List of ints or floats of length n
        Result list.

    """
    return [l2[i] if not l1[i] else l1[i] for i in range(len(l1))]

def extract_points(session, seq, x, y, z, pt, flg_proj=False, lims=False):
    """Extract the positions of the robot in a brick given its spatial delimitation

    Parameters
    ----------
    rosmsg : ROSMsg subclass instance
        The SQLAlcheny ORM Base instance to communicate with the corresponding table
        in the Postgresql database.
    session : sqlalchemy.orm.session.Session
        SQLAlchemy Session instance to send queries with.
    xmin, xmax, ymin, ymax, altmin, altmax : float
        Spatial boundaries of the brick to extract.

    Returns
    -------
    points : ndarray, (N,3)
        Positions of the robot in the given brick.
    seq : SQLAlchemy Column
        Column of primary keys of the selected points.
    l1 : List of ints or floats
        List of the limits wrt x and y axis of the selected data.
    """
    if lims is not False :
        xmin, xmax, ymin, ymax, zmin, zmax = lims
    else :
        xmin, xmax, ymin, ymax, zmin, zmax = False, False, False, False, False, False
    xtotmin, xtotmax, ytotmin, ytotmax, ztotmin, ztotmax = session.query(\
    func.min(x),func.max(x),\
    func.min(y),func.max(y),\
    func.min(z),func.max(z)).one()
    # xtotmin, xtotmax, ytotmin, ytotmax, ztotmin, ztotmax = [-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]
    print(xmin,xmax, ymin, ymax)
    if flg_proj:
        xmin,ymin = proj_xy(xmin, ymin)
        xmax,ymax = proj_xy(xmax, ymax)
    # l1 = [xmin,xmax,ymin,ymax]
    # l2 = [zmin, zmax]
    # l3 = [xtotmin,xtotmax,ytotmin,ytotmax]
    # l4 = [ztotmin, ztotmax]
    # l1 = [l3[i] if not l1[i] else l1[i] for i in range(len(l1))]
    # l2 = [l4[i] if not l2[i] else l2[i] for i in range(len(l2))]
    l1 = smallest_list([xmin,xmax,ymin,ymax], [xtotmin,xtotmax,ytotmin,ytotmax])
    l2 = smallest_list([zmin, zmax], [ztotmin, ztotmax])
    print("\n\n\n",l1, l2, zmin, zmax, ztotmin, ztotmax , "\n\n\n")
    l1 = [float(i) for i in l1]
    l2 = [float(i) for i in l2]
    xmin,xmax,ymin,ymax = l1
    zmin, zmax = l2
    print("\n\n\n",l1, l2, "\n\n\n")
    rect = create_rect(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    sub0 = session.query(seq, ST_AsText(pt), z).\
            filter(ST_Contains(rect,pt)).\
            filter(z>zmin).\
            filter(z<zmax).\
            all()
    sub0 = np.array(sub0)
    print("\n\n\n",sub0 , "\n\n\n")
    seq = sub0[:,0]
    sub = sub0[:,1:]
    nb_msgs = len(sub)
    points = np.zeros([nb_msgs, 3])
    for i in range(nb_msgs):
        points[i,:] = float(sub[i,0][6:-1].split(" ")[0]), float(sub[i,0][6:-1].split(" ")[1]), sub[i,1]
    l1
    return points, seq, [xmin,xmax,ymin,ymax,zmin,zmax]
