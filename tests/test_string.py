def get_col_spec(scale,precision, x,y):
    s = "{:0%s.%sf}" % (scale-precision,precision)
    print(s)
    s2 = ("POINT({},{})".format(s,s)).format(x,y)
    return s2

print(get_col_spec(10,2,2.369,456.39))
