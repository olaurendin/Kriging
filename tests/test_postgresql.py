import psycopg2 as ps

conn = ps.connect(dbname="opengeo",host="localhost",user="ganlan",password="88a002d51c1")
print("ok")
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS test;")
c.execute("CREATE TABLE test (\
id INTEGER NOT NULL PRIMARY KEY\
)")
conn.commit()
