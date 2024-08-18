<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Projet d'Analyse de Données de Chicago</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #2c3e50;
        }
        code {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 2px 5px;
            font-family: 'Courier New', Courier, monospace;
        }
        pre {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 10px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>Chicago Data Analysis Project</h1>
    <p>Ce projet utilise Python avec les bibliothèques pandas et sqlite3 pour effectuer une analyse de données sur plusieurs ensembles de données liés à Chicago, notamment les données du recensement, les écoles publiques de Chicago et les données sur la criminalité à Chicago.</p>
    
    <h2>Données</h2>
    <p>Les ensembles de données utilisés dans ce projet sont les suivants :</p>
    <ol>
        <li><code>ChicagoCensusData.csv</code>: fichier CSV contenant Données du recensement de Chicago.</li>
        <li><code>ChicagoPublicSchools.csv</code>: fichier CSV contenant Données sur les écoles publiques de Chicago.</li>
        <li><code>ChicagoCrimeData.csv</code>: fichier CSV contenant Données sur la criminalité à Chicago.</li>
    </ol>

    <h2>Exemples de Requêtes SQL</h2>
    <ol>
        <li>
            <p>Nombre total d'incidents criminels dans la table3 :</p>
            <pre><code>SELECT COUNT(ID) FROM table3</code></pre>
        </li>
        <li>
            <p>Zones communautaires et noms associés où le revenu par habitant est inférieur à 11 000 :</p>
            <pre><code>SELECT COMMUNITY_AREA_NUMBER , COMMUNITY_AREA_NAME  FROM table1 WHERE PER_CAPITA_INCOME < 11000</code></pre>
        </li>
        <li>
            <p>Nombre d'incidents criminels et dates où l'incident a eu lieu au cours des 18 dernières années :</p>
            <pre><code>SELECT CASE_NUMBER, DATE FROM table3 WHERE (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18</code></pre>
        </li>
        <li>
            <p>Informations sur la structure de la table3 :</p>
            <pre><code>PRAGMA table_info(table3)</code></pre>
        </li>
        <li>
            <p>Incidents de kidnapping survenus au cours des 18 dernières années :</p>
            <pre><code>SELECT * FROM table3 WHERE PRIMARY_TYPE = 'KIDNAPPING' AND (strftime('%Y', 'now') - strftime('%Y', DATE)) < 18</code></pre>
        </li>
        <li>
            <p>Types de criminalité (PRIMARY_TYPE) et noms d'écoles associés (NAME_OF_SCHOOL) :</p>
            <pre><code>SELECT t.PRIMARY_TYPE, x.NAME_OF_SCHOOL FROM table3 t JOIN table2 x ON t.LOCATION = x.LOCATION</code></pre>
        </li>
    </ol>

    <h2>Analyses Supplémentaires</h2>
    <p>t</p>

    <h2>Licence</h2>
    <p>Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.</p>

    <h2>Auteur</h2>
    <p>ZAKARIAE YAHYA</p>
</body>
</html>
