import pyproj

"""######################### PROJECTION CONSTANTS #########################"""

epsg_inproj = 'epsg:4326'       # GPS projection
epsg_outproj = 'epsg:2154'      # Lambert93 projection

"""######################### PROJECTION FUNCTIONS #########################"""

def inProj():
    """ Input projection of the coordinates (default : GPS)
    Reference : http://spatialreference.org/ref/epsg/4326/
    """
    return pyproj.Proj(init=epsg_inproj)

def outProj():
    """ Output projection of the coordinates (default : Lambert93)
    Reference : http://spatialreference.org/ref/epsg/rgf93-lambert-93/
    """
    return pyproj.Proj(init=epsg_outproj)

def proj_xy(lon, lat):
    return pyproj.transform(inProj(),outProj(),lon,lat)

def proj_x(lon,lat):
    return proj_xy(lon, lat)[0]

def proj_y(lon,lat):
    return proj_xy(lon, lat)[1]

"""######################### PERSONNALIZED FUNCTIONS #########################"""

def input_pt(var1, var2):
    return 'POINT({} {})'.format(var1,var2)

def input_pt_proj(var1, var2):
    return 'POINT({} {})'.format(proj_x(var1,var2), proj_y(var1,var2))
