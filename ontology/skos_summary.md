# SKOS Taxonomies Summary

## 1. genres_skos.ttl

Contains a hierarchical SKOS taxonomy with **17 genres** (all unique genres in dataset) organized in 4 broader categories:

| Category | Genres |
|----------|--------|
| **Action & Adventure** | Action (108), Adventure (107), Thriller (52), War (7), Western (3) |
| **Drama** | Drama (53), Crime (29), Romance (10), History (3) |
| **Speculative** | Science Fiction (74), Fantasy (47), Mystery (15), Horror (5) |
| **Other** | Family (39), Comedy (36), Animation (25), Music (1) |

---

## 2. keywords_skos.ttl

Contains a hierarchical SKOS taxonomy with **30 keywords** organized in 5 broader categories:

| Category | Keywords |
|----------|----------|
| **Production** | 3D (35), AfterCreditsStinger (33), DuringCreditsStinger (32), BasedOnComic (27), BasedOnNovel (21), Sequel (18), IMAX (13), BasedOnYoungAdultNovel (9) |
| **Theme** | Magic (15), Violence (12), Friendship (11), Revenge (11), Murder (9) |
| **Content** | Superhero (32), Dystopia (28), MarvelComic (26), MarvelCinematicUniverse (13), Alien (11), SuperPowers (9), ArtificialIntelligence (8), Hero (7), Mutant (7) |
| **Setting** | Space (11), SpaceOpera (11), Spaceship (9), Future (8), NewYorkCity (7) |
| **Character** | FemaleProtagonist (8), Wizard (8), TimeTravel (7) |

---

## Extracted Data Files

| File | Description |
|------|-------------|
| `extracted_genres.csv` | Top 30 genres with ID, name, count |
| `extracted_keywords.csv` | Top 30 keywords with ID, name, count |

---

## SKOS Features Used

| Feature | Description |
|---------|-------------|
| `skos:ConceptScheme` | Defines the taxonomy |
| `skos:Concept` | Individual concepts |
| `skos:prefLabel` / `skos:altLabel` | Preferred and alternate labels |
| `skos:definition` | Concept definitions |
| `skos:broader` / `skos:narrower` | Hierarchical relationships |
| `skos:related` | Associative relationships |
| `skos:notation` | Original IDs from the dataset |
