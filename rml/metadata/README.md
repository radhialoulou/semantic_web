# RML Transformation - Movies Ontology

## Structure des fichiers

```
rml/metadata
├── movies_mapping.rml.ttl    # Mapping RML principal
├── normalize_movies.py       # Script de normalisation CSV
├── run_rml.ps1              # Script d'exécution PowerShell
├── movies_normalized.csv     # Données films (généré)
├── movies_genres.csv         # Relations film-genre (généré)
├── movies_companies.csv      # Relations film-société (généré)
├── movies_countries.csv      # Relations film-pays (généré)
└── movies_collections.csv    # Relations film-collection (généré)
```

## Prérequis

1. **Python 3.x** avec pandas
2. **Java 11+** pour RMLMapper
3. **RMLMapper** : [Télécharger ici](https://github.com/RMLio/rmlmapper-java/releases)

## Utilisation

### Étape 1 : Normaliser les données
```bash
python normalize_movies.py
```

### Étape 2 : Exécuter RMLMapper
```bash
java -jar rmlmapper.jar -m movies_mapping.rml.ttl -o movies_output.ttl -s turtle
```

### Ou en une commande (PowerShell)
```powershell
.\run_rml.ps1
```

## Mapping vers l'ontologie

| Source CSV | Classe Ontologie | Propriétés |
|------------|------------------|------------|
| movies_normalized.csv | `movie:Movie` | hasTitle, hasTagline |
| movies_normalized.csv | `movie:MovieInformation` | hasBudget, hasReleaseDate, hasRuntime, hasLanguage |
| movies_normalized.csv | `movie:MovieResult` | hasVoteAverage, hasVoteCount, hasRevenue |
| movies_normalized.csv | `movie:MovieContent` | hasPlot |
| movies_genres.csv | - | movie:hasGenre → genres:* (SKOS) |
| movies_companies.csv | `foaf:Organization` | foaf:name |
| movies_countries.csv | - | movie:hasProductionCountry |
| movies_collections.csv | `movie:Collection` | collectionName, belongsToCollection |

## Lien avec SKOS Genres

Les genres sont mappés vers les concepts SKOS définis dans `ontology/genres_skos.ttl` :

| Genre ID | Genre Name | URI SKOS |
|----------|------------|----------|
| 28 | Action | genres:Action |
| 12 | Adventure | genres:Adventure |
| 878 | Science Fiction | genres:ScienceFiction |
| 18 | Drama | genres:Drama |
| 53 | Thriller | genres:Thriller |
| 80 | Crime | genres:Crime |
| 14 | Fantasy | genres:Fantasy |
| 35 | Comedy | genres:Comedy |
| ... | ... | ... |

## Exemple de triplets générés

```turtle
@prefix movie: <http://saraaymericradhi.org/movie-ontology#> .
@prefix genres: <http://saraaymericradhi.org/movie-ontology/genres#> .

<http://saraaymericradhi.org/movie-ontology/movie/27205>
    a movie:Movie ;
    movie:hasTitle "Inception"^^xsd:string ;
    movie:hasTagline "Your mind is the scene of the crime."^^xsd:string ;
    movie:hasInformation <http://saraaymericradhi.org/movie-ontology/movie/27205/information> ;
    movie:hasResult <http://saraaymericradhi.org/movie-ontology/movie/27205/result> ;
    movie:hasContent <http://saraaymericradhi.org/movie-ontology/movie/27205/content> .

<http://saraaymericradhi.org/movie-ontology/movie/27205/content>
    a movie:MovieContent ;
    movie:hasGenre genres:Action, genres:Thriller, genres:ScienceFiction, 
                   genres:Mystery, genres:Adventure .
```
