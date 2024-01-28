# Chicago Data Analysis Project

Ce projet utilise Python avec les bibliothèques pandas et sqlite3 pour effectuer une analyse de données sur plusieurs ensembles de données liés à Chicago, notamment les données du recensement, les écoles publiques de Chicago et les données sur la criminalité à Chicago.

## Configuration
Assurez-vous d'avoir Python installé sur votre machine, ainsi que les bibliothèques nécessaires. Vous pouvez installer ces bibliothèques en utilisant la commande suivante :

```bash
pip install pandas sqlite3
## Données

Les ensembles de données utilisés dans ce projet sont les suivants :

1. [ChicagoCensusData.csv]: fichier CSV contenant Données du recensement de Chicago.
2. [ChicagoPublicSchools.csv]:fichier CSV contenant Données sur les écoles publiques de Chicago.
3. [ChicagoCrimeData.csv]: fichier CSV contenant Données sur la criminalité à Chicago.

## Exemples de Requêtes SQL
1. Nombre total d'incidents criminels dans la table3 :
    SELECT COUNT(ID) FROM table3
2. Zones communautaires et noms associés où le revenu par habitant est inférieur à 11 000 :
    SELECT COMMUNITY_AREA_NUMBER , COMMUNITY_AREA_NAME  FROM table1 WHERE PER_CAPITA_INCOME < 11000
3. Nombre d'incidents criminels et dates où l'incident a eu lieu au cours des 18 dernières années :
    SELECT CASE_NUMBER, DATE FROM table3 WHERE (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18
4. Informations sur la structure de la table3 :
    PRAGMA table_info(table3)
5. Incidents de kidnapping survenus au cours des 18 dernières années :
    SELECT * FROM table3 WHERE PRIMARY_TYPE = 'KIDNAPPING' AND (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18
6. Types de criminalité (PRIMARY_TYPE) et noms d'écoles associés (NAME_OF_SCHOOL) :
    SELECT t.PRIMARY_TYPE, x.NAME_OF_SCHOOL FROM table3 t JOIN table2 x ON t.LOCATION = x.LOCATION

## Analyses Supplémentaires
Explorez davantage les données en adaptant les requêtes à vos besoins spécifiques. Voici quelques idées d'analyses supplémentaires que vous pourriez effectuer :
1. Analyse de la Sécurité des Écoles
2. Comparaison des Niveaux de Revenu par Zone Communautaire
3. Évolution des Incidents Criminels au Fil du Temps
## Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.
## Auteur
ZAKARIAE YAHYA 

