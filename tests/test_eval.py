import pyproj


def inProj():
    """ Input projection of the coordinates (GPS)
    Reference : http://spatialreference.org/ref/epsg/4326/
    """
    return pyproj.Proj(init='epsg:4326')

def outProj():
    """ Output projection of the coordinates (Lambert93)
    Reference : http://spatialreference.org/ref/epsg/rgf93-lambert-93/
    """
    return pyproj.Proj(init='epsg:2154')

def proj_x(lon,lat):
    return pyproj.transform(inProj(),outProj(),lon,lat)[0]

def proj_y(lon,lat):
    return pyproj.transform(inProj(),outProj(),lon,lat)[1]

if __name__ == "__main__":
    lon = -4.491780568333334
    lat = 48.378888305

    x,y=pyproj.transform(inProj(),outProj(),lon,lat)

    ### First try
    # print(eval("'({},{}) ok'.format(x,y)", {"x":x, "y":y}))
    print(eval('(proj_x(lat,lon),proj_y(lat,lon))', {"proj_x":proj_x, "proj_y":proj_y, "lat":lat, "lon":lon}))
    print(eval("eval('(proj_x(lat,lon),proj_y(lat,lon))', {'proj_x':proj_x, 'proj_y':proj_y, 'lat':lat, 'lon':lon})"))

    ###Second try
    str0 = "'POINT({},{})'.format(x,y)"
    strproj = "{'x':proj_x(lon, lat), 'y':proj_y(lon, lat)}"
    strnoproj = "{'x':lon, 'y':lat}"
    print(eval("'POINT({},{})'.format(x,y)", {'x':proj_x(lon, lat), 'y':proj_y(lon, lat)}))
    print(eval(str0, eval(strproj, {"proj_x":proj_x, "proj_y":proj_y, "lon":lon, "lat":lat})))
    print(eval(str0, eval(strnoproj, {"proj_x":proj_x, "proj_y":proj_y, "lon":lon, "lat":lat})))

    ### Third try
    str0 = "'POINT({},{})'.format(x,y)"
    def strproj(x1, x2):
        return "{'x':proj_x(%(x1)s, %(x2)s), 'y':proj_y(%(x1)s, %(x2)s)}" % {"x1": x1, "x2": x2}
    def strnoproj(x1, x2):
        return "{'x':%s, 'y':%s}" %(x1, x2)

    print(strproj("1", "2"))
    print(strnoproj("1", "2"))
    print(eval(str0, eval(strproj("lon", "lat"), {"proj_x":proj_x, "proj_y":proj_y, "lon":lon, "lat":lat})))
    print(eval(str0, eval(strnoproj("lon", "lat"), {"proj_x":proj_x, "proj_y":proj_y, "lon":lon, "lat":lat})))



    ### Final implementation
    str_point = "'POINT({},{})'.format(var1,var2)"
    def strproj(x1, x2):
        return "{'var1':proj_x(%(x1)s, %(x2)s), 'var2':proj_y(%(x1)s, %(x2)s)}" % {"x1": x1, "x2": x2}
    def strnoproj(x1, x2):
        return "{'var1':%s, 'var2':%s}" %(x1, x2)

    def fake_eval(str1, str2):
        dict_columns = {"proj_x":proj_x, "proj_y":proj_y, "lon":lon, "lat":lat}
        return eval(str1, eval(str2, dict_columns))

    print(fake_eval(str_point, strproj("lon", "lat")))
    print(fake_eval(str_point, strnoproj("lon", "lat")))




















# EOF
