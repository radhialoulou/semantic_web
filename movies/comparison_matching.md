# Comparaison des approches de Matching : Fuzzy Matching vs Embeddings

Ce document compare deux méthodes pour lier les films de notre dataset principal (`movies_metadata_top200.csv`) avec les résumés détaillés issus de Wikipedia (`wiki_movie_plots_deduped.csv`).

## 1. Approche "Fuzzy Title Match" (Normalisation)
**Méthode :** 
- Nettoyage des titres (minuscules, suppression ponctuation).
- Création d'un dictionnaire indexé par titre normalisé.
- Correspondance exacte sur le titre normalisé.
- En cas d'ambiguïté (plusieurs films avec le même titre), filtre sur l'année de sortie (+/- 1 an).

**Avantages :** es
- Extrêmement rapide (O(1) avec le dictionnaire).
- Pas de dépendances lourdes (juste Python standard/Pandas).
- Très précis pour les titres identiques (ex: "Avatar" = "Avatar").

**Inconvénients :**
- **Très sensible aux variations** : "Star Wars" ne matche pas avec "Star Wars Episode IV: A New Hope".
- Rate les sous-titres ou les formatages différents (ex: "Seven" vs "Se7en").
- Taux de correspondance plus faible sur des datasets hétérogènes.

---

## 2. Approche Sémantique (Embeddings)
**Méthode :**
- Utilisation de **Sentence-BERT** (`all-MiniLM-L6-v2`) pour transformer les titres en vecteurs denses.
- Calcul de la **similarité Cosine** entre chaque titre du Top 200 et *tous* les titres Wiki.
- Sélection du candidat avec le meilleur score, sous condition :
  1. **Filtre Année** : La sortie doit être à +/- 1 an (Crucial pour les suites).
  2. **Seuil de confiance** : Score > 0.75 (pour éviter les faux positifs).

**Avantages :**
- **Robustesse sémantique** : Capte "Harry Potter 1" et "Harry Potter and the Sorcerer's Stone" comme étant le même film.
- Gère la ponctuation et les variations mineures invisibles pour un humain.
- Taux de correspondance élevé (~91.5% soit 183/200 films).

**Inconvénients :**
- Plus lent (nécessite d'encoder ~35k titres Wiki).
- Nécessite un GPU ou un CPU décent.
- **Risque de Faux Positifs** : Si le seuil est trop bas (< 0.6), il peut lier "Brave" à "Courageous" (sens proche) ou des films n'ayant rien à voir mais sortis la même année.

## Conclusion & Choix
Nous avons retenu l'approche par **Embeddings** car elle permet de récupérer des films majeurs (*Star Wars*, sagas Harry Potter, etc.) que l'approche stricte rejetait. Avec un filtrage strict sur l'année, elle offre le meilleur compromis couverture/précision.
