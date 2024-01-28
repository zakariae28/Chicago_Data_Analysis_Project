import pandas as pd
import sqlite3

# Chemins des fichiers CSV
chemin_csv_recensement = r'D:\bureau\BD&AI 1\BD\with python\projet\ChicagoCensusData.csv'
chemin_csv_ecoles = r'D:\bureau\BD&AI 1\BD\with python\projet\ChicagoPublicSchools.csv'
chemin_csv_criminalite = r'D:\bureau\BD&AI 1\BD\with python\projet\ChicagoCrimeData.csv'

# Chargement des données CSV
df_recensement = pd.read_csv(chemin_csv_recensement, delimiter=',')
df_ecoles = pd.read_csv(chemin_csv_ecoles, delimiter=',')
df_criminalite = pd.read_csv(chemin_csv_criminalite, delimiter=',')

# Connexion à la base de données SQLite
conn = sqlite3.connect('FinalDB.db')

# Suppression des tables si elles existent
conn.execute('DROP TABLE IF EXISTS table_recensement')
conn.execute('DROP TABLE IF EXISTS table_ecoles')
conn.execute('DROP TABLE IF EXISTS table_criminalite')

# Enregistrement des DataFrames dans des tables SQLite
df_recensement.to_sql('table_recensement', conn)
df_ecoles.to_sql('table_ecoles', conn)
df_criminalite.to_sql('table_criminalite', conn)

# Exécution des requêtes SQL
requete_nombre_criminalite = 'SELECT COUNT(ID) FROM table_criminalite'
resultat_nombre_criminalite = pd.read_sql(requete_nombre_criminalite, conn)
print(resultat_nombre_criminalite)

requete_zones_faible_revenu = 'SELECT COMMUNITY_AREA_NUMBER, COMMUNITY_AREA_NAME FROM table_recensement WHERE PER_CAPITA_INCOME < 11000'
resultat_zones_faible_revenu = pd.read_sql(requete_zones_faible_revenu, conn)
print(resultat_zones_faible_revenu)

requete_criminalite_recente = "SELECT CASE_NUMBER, DATE FROM table_criminalite WHERE (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18"
resultat_criminalite_recente = pd.read_sql(requete_criminalite_recente, conn)
print(resultat_criminalite_recente)

requete_info_table_criminalite = "PRAGMA table_info(table_criminalite)"
resultat_info_table_criminalite = pd.read_sql(requete_info_table_criminalite, conn)
print(resultat_info_table_criminalite)

requete_kidnapping_recent = "SELECT * FROM table_criminalite WHERE PRIMARY_TYPE = 'KIDNAPPING' AND (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18"
resultat_kidnapping_recent = pd.read_sql(requete_kidnapping_recent, conn)
print(resultat_kidnapping_recent)

requete_relation_criminalite_ecole = "SELECT t.PRIMARY_TYPE, x.NAME_OF_SCHOOL FROM table_criminalite t JOIN table_ecoles x ON t.LOCATION = x.LOCATION"
resultat_relation_criminalite_ecole = pd.read_sql(requete_relation_criminalite_ecole, conn)
print(resultat_relation_criminalite_ecole)

requete_moyenne_score_securite = "SELECT `Elementary, Middle, or High School`, AVG(SAFETY_SCORE) AS avg_safety_score FROM table_ecoles WHERE `Elementary, Middle, or High School` IN ('MS', 'HS', 'ES') GROUP BY `Elementary, Middle, or High School`"
resultat_moyenne_score_securite = pd.read_sql(requete_moyenne_score_securite, conn)
print(resultat_moyenne_score_securite)

requete_pauvrete_revenu = "SELECT COMMUNITY_AREA_NUMBER, PER_CAPITA_INCOME FROM table_recensement ORDER BY PERCENT_HOUSEHOLDS_BELOW_POVERTY DESC LIMIT 5"
resultat_pauvrete_revenu = pd.read_sql(requete_pauvrete_revenu, conn)
print(resultat_pauvrete_revenu)

requete_zone_criminalite_max = "SELECT v.COMMUNITY_AREA_NUMBER, v.COMMUNITY_AREA_NAME, COUNT(b.COMMUNITY_AREA_NUMBER) as crime_count FROM table_recensement v LEFT JOIN table_criminalite b ON v.COMMUNITY_AREA_NUMBER = b.COMMUNITY_AREA_NUMBER GROUP BY v.COMMUNITY_AREA_NUMBER, v.COMMUNITY_AREA_NAME ORDER BY crime_count DESC LIMIT 1"
resultat_zone_criminalite_max = pd.read_sql(requete_zone_criminalite_max, conn)
print(resultat_zone_criminalite_max)

requete_zone_hardship_max = "SELECT COMMUNITY_AREA_NAME FROM table_recensement WHERE HARDSHIP_INDEX = (SELECT MAX(HARDSHIP_INDEX) FROM table_recensement)"
resultat_zone_hardship_max = pd.read_sql(requete_zone_hardship_max, conn)
print(resultat_zone_hardship_max)

requete_zone_criminalite_max_comm = "SELECT COMMUNITY_AREA_NAME FROM table_recensement WHERE COMMUNITY_AREA_NUMBER = (SELECT COMMUNITY_AREA_NUMBER FROM table_criminalite GROUP BY COMMUNITY_AREA_NUMBER ORDER BY COUNT(*) DESC LIMIT 1)"
resultat_zone_criminalite_max_comm = pd.read_sql(requete_zone_criminalite_max_comm, conn)
print(resultat_zone_criminalite_max_comm)
