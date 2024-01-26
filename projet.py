import pandas as pd 
import sqlite3
chemin_du_fichier_csv1 = r'D:\bureau\BD&AI 1\BD\with python\projet\ChicagoCensusData.csv'
chemin_du_fichier_csv2 = r'D:\bureau\BD&AI 1\BD\with python\projet\ChicagoPublicSchools.csv'
chemin_du_fichier_csv3 = r'D:\bureau\BD&AI 1\BD\with python\projet\ChicagoCrimeData.csv'
data1 = pd.read_csv(chemin_du_fichier_csv1, delimiter=',')
data2 = pd.read_csv(chemin_du_fichier_csv2, delimiter=',')
data3 = pd.read_csv(chemin_du_fichier_csv3, delimiter=',')
conn = sqlite3.connect('FinalDB.db')
# Drop tables if they exist
conn.execute('DROP TABLE IF EXISTS table1')
conn.execute('DROP TABLE IF EXISTS table2')
conn.execute('DROP TABLE IF EXISTS table3')
data1.to_sql('table1', conn)
data2.to_sql('table2', conn)
data3.to_sql('table3', conn)
query = 'SELECT COUNT(ID) FROM table3'
df = pd.read_sql(query, conn)
print(df)
a='''SELECT COMMUNITY_AREA_NUMBER , COMMUNITY_AREA_NAME  FROM table1 WHERE PER_CAPITA_INCOME < 11000'''
df = pd.read_sql(a, conn)
print(df)
b = ''' SELECT CASE_NUMBER,DATE FROM table3 WHERE (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18 '''
df = pd.read_sql(b, conn)
print(df)
table_name = 'table3'
c = f"PRAGMA table_info({table_name})"
df_info = pd.read_sql(c, conn)
print(df_info)
d='''SELECT * FROM table3 WHERE PRIMARY_TYPE = 'KIDNAPPING' AND (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18 '''
ki = pd.read_sql(d, conn)
print(ki)
e = '''SELECT t.PRIMARY_TYPE, x.NAME_OF_SCHOOL FROM table3 t JOIN table2 x ON t.LOCATION = x.LOCATION '''
kii = pd.read_sql(e, conn)
print(kii)
f = '''SELECT `Elementary, Middle, or High School`, AVG(SAFETY_SCORE) AS avg_safety_score FROM table2 WHERE `Elementary, Middle, or High School` IN ('MS', 'HS', 'ES') GROUP BY `Elementary, Middle, or High School` '''
kiii=pd.read_sql(f,conn)
print(kiii) 
g=''' SELECT COMMUNITY_AREA_NUMBER,PER_CAPITA_INCOME FROM table1 ORDER BY PERCENT_HOUSEHOLDS_BELOW_POVERTY DESC LIMIT 5'''
kiiii = pd.read_sql(g, conn)
print(kiiii)
g = '''SELECT v.COMMUNITY_AREA_NUMBER, v.COMMUNITY_AREA_NAME, COUNT(b.COMMUNITY_AREA_NUMBER) as crime_count FROM table1 v LEFT JOIN table3 b ON v.COMMUNITY_AREA_NUMBER = b.COMMUNITY_AREA_NUMBER GROUP BY v.COMMUNITY_AREA_NUMBER, v.COMMUNITY_AREA_NAME ORDER BY crime_count DESC LIMIT 1'''
result = pd.read_sql(g, conn)
print(result)
h='''SELECT COMMUNITY_AREA_NAME FROM table1 WHERE HARDSHIP_INDEX =(SELECT MAX(HARDSHIP_INDEX) FROM table1)'''
result = pd.read_sql(h, conn)
print(result)
i = ''' SELECT COMMUNITY_AREA_NAME FROM table1 WHERE COMMUNITY_AREA_NUMBER = ( SELECT COMMUNITY_AREA_NUMBER FROM table3 GROUP BY COMMUNITY_AREA_NUMBER ORDER BY COUNT(*) DESC LIMIT 1)'''
result = pd.read_sql(i, conn)
print(result)
