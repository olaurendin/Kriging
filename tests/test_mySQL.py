# import _mysql as ms
import MySQLdb as ms
import os
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

def printtable(c, table):
    c.execute("DESCRIBE {};".format(str(table)))
    l = DataFrame(np.array(c.fetchall()), columns=["Field", "Type", "Null", "Key", "Default", "Extra"])
    print(l)

# os.system("mysql-workbench --force-opengl-render")
# os.system("systemctl status mysql")
# os.system("sudo systemcl start mysql")
db = ms.connect(host="localhost", user = "root", passwd="88a002d51c1", db="test")

c = db.cursor()
c.execute("USE test;")
c.execute("DROP TABLE IF EXISTS tabletest;")
# mdata_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,\
c.execute("CREATE TABLE tabletest(\
mdata_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,\
t BIGINT NOT NULL,\
X_Axis DECIMAL(21,20) NOT NULL,\
Y_Axis DECIMAL(21,20) NOT NULL,\
Z_Axis DECIMAL(21,20) NOT NULL,\
Absolute_Value DECIMAL(21,20) NOT NULL\
) ENGINE = MYISAM;")

pathstr = "data/Boatbot-exp/117/boatbot_exp117"
filestr = "20180711094658-boatbot-0000.csv"
filewstr = "20180711094658-boatbot-0000_2.csv"
file = os.path.join(pathstr, filestr)
filew = os.path.join(pathstr, filewstr)
nblines = 13

# f=pd.DataFrame(pd.read_csv(file, header=nblines,names=["t", "X-Axis", "Y-Axis", "Z-Axis", "Absolute_Value"], sep=";", decimal = ","))
# f.to_csv(filew,sep = ";", decimal = ".", columns=["t", "X-Axis", "Y-Axis", "Z-Axis", "Absolute_Value"])
# print(f)

# ENCLOSED BY '"'
# CREATE TABLE geom (g GEOMETRY);
# ALTER TABLE geom ADD pt POINT;
# ALTER TABLE geom DROP pt;

# UPDATE myTable
# SET Coord = PointFromText(CONCAT('POINT(',myTable.DLong,' ',myTable.DLat,')'));

# LOAD DATA INFILE 'file.txt'
# INTO TABLE t1
# (column1, @var1)
# SET column2 = @var1/100;

# CREATE TABLE `table_with_a_point` (
# `id` bigint(20) not null,
# `location` point not NULL,
# `latitude` float default NULL,
# `longitude` float default NULL,
# `value` int(11) not null,
# PRIMARY KEY (`id`)
# );
# create spatial index table_with_a_point_index on table_with_a_point(location);
#
# LOAD DATA LOCAL INFILE 'somedata.txt'
# INTO TABLE table_with_a_point
# COLUMNS TERMINATED BY ' ' LINES TERMINATED BY '\r\n'
# (id, latitude, longitude, value)
# set location = PointFromText(CONCAT('POINT(',latitude,' ',longitude,')'));

# SELECT AsText(g) FROM geom;

c.execute("LOAD DATA LOCAL INFILE '{}' \
INTO TABLE tabletest \
columns TERMINATED BY ';' \
LINES TERMINATED BY '\n' \
IGNORE {} LINES \
(t, @X_Axis, @Y_Axis, @Z_Axis, @Absolute_Value) \
SET X_Axis = replace(@X_Axis, ',', '.'), \
Y_Axis = replace(@Y_Axis, ',', '.'), \
Z_Axis = replace(@Z_Axis, ',', '.'), \
Absolute_Value = replace(@Absolute_Value, ',', '.');".format(file, nblines))
c.execute("SELECT * FROM tabletest WHERE t<={};".format(1531302418418))
# print(c.fetchall())
printtable(c, "tabletest")
