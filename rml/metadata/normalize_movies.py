"""
Script de normalisation des données movies_metadata_top200.csv
pour la transformation RML.

Ce script extrait les colonnes JSON imbriquées et crée des fichiers CSV normalisés:
- movies_normalized.csv : données principales des films
- movies_genres.csv : relation films <-> genres
- movies_companies.csv : relation films <-> sociétés de production  
- movies_countries.csv : relation films <-> pays de production
- movies_collections.csv : relation films <-> collections

Usage: python normalize_movies.py
"""

import pandas as pd
import ast
import os
import re

# Chemins
INPUT_FILE = "../../movies/movies_metadata_top200.csv"
OUTPUT_DIR = "./"

# Mapping des IDs de genres vers les noms SKOS (correspondant à genres_skos.ttl)
GENRE_ID_TO_SKOS = {
    28: "Action",
    12: "Adventure", 
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    18: "Drama",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "ScienceFiction",
    53: "Thriller",
    10752: "War",
    37: "Western",
    10751: "Family"
}


def safe_eval(value):
    """Évalue une chaîne Python de manière sécurisée."""
    if pd.isna(value) or value == "" or value == "[]":
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def safe_eval_dict(value):
    """Évalue une chaîne Python dict de manière sécurisée."""
    if pd.isna(value) or value == "":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def clean_runtime(value):
    """Nettoie la valeur runtime en supprimant le .0"""
    if pd.isna(value):
        return ""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return ""


def clean_vote_count(value):
    """Nettoie vote_count en entier."""
    if pd.isna(value):
        return ""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return ""


def normalize_movies():
    """Fonction principale de normalisation."""
    
    print(f"Lecture de {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  -> {len(df)} films chargés")
    
    # =========================================================================
    # 1. Créer movies_normalized.csv (données principales)
    # =========================================================================
    print("\n1. Création de movies_normalized.csv...")
    
    movies_norm = df[[
        'id', 'imdb_id', 'title', 'original_title', 'original_language',
        'overview', 'tagline', 'budget', 'revenue', 'runtime',
        'release_date', 'vote_average', 'vote_count', 'popularity',
        'status', 'adult', 'homepage', 'poster_path'
    ]].copy()
    
    # Nettoyer runtime et vote_count
    movies_norm['runtime'] = movies_norm['runtime'].apply(clean_runtime)
    movies_norm['vote_count'] = movies_norm['vote_count'].apply(clean_vote_count)
    
    # Nettoyer revenue (enlever .0)
    movies_norm['revenue'] = movies_norm['revenue'].apply(
        lambda x: int(float(x)) if pd.notna(x) and x != "" else ""
    )
    
    movies_norm.to_csv(os.path.join(OUTPUT_DIR, "movies_normalized.csv"), index=False)
    print(f"  -> {len(movies_norm)} films exportés")
    
    # =========================================================================
    # 2. Créer movies_genres.csv
    # =========================================================================
    print("\n2. Création de movies_genres.csv...")
    
    genres_rows = []
    for _, row in df.iterrows():
        movie_id = row['id']
        genres = safe_eval(row['genres'])
        for genre in genres:
            genre_id = genre.get('id')
            genre_name = genre.get('name')
            # Mapper vers le nom SKOS
            skos_name = GENRE_ID_TO_SKOS.get(genre_id, genre_name.replace(" ", ""))
            genres_rows.append({
                'movie_id': movie_id,
                'genre_id': genre_id,
                'genre_name': skos_name
            })
    
    genres_df = pd.DataFrame(genres_rows)
    genres_df.to_csv(os.path.join(OUTPUT_DIR, "movies_genres.csv"), index=False)
    print(f"  -> {len(genres_df)} relations film-genre exportées")
    
    # =========================================================================
    # 3. Créer movies_companies.csv
    # =========================================================================
    print("\n3. Création de movies_companies.csv...")
    
    companies_rows = []
    for _, row in df.iterrows():
        movie_id = row['id']
        companies = safe_eval(row['production_companies'])
        for company in companies:
            companies_rows.append({
                'movie_id': movie_id,
                'company_id': company.get('id'),
                'company_name': company.get('name')
            })
    
    companies_df = pd.DataFrame(companies_rows)
    # Supprimer les doublons de companies (garder une seule définition par company)
    companies_df.to_csv(os.path.join(OUTPUT_DIR, "movies_companies.csv"), index=False)
    print(f"  -> {len(companies_df)} relations film-société exportées")
    
    # =========================================================================
    # 4. Créer movies_countries.csv
    # =========================================================================
    print("\n4. Création de movies_countries.csv...")
    
    countries_rows = []
    for _, row in df.iterrows():
        movie_id = row['id']
        countries = safe_eval(row['production_countries'])
        for country in countries:
            countries_rows.append({
                'movie_id': movie_id,
                'country_code': country.get('iso_3166_1'),
                'country_name': country.get('name')
            })
    
    countries_df = pd.DataFrame(countries_rows)
    countries_df.to_csv(os.path.join(OUTPUT_DIR, "movies_countries.csv"), index=False)
    print(f"  -> {len(countries_df)} relations film-pays exportées")
    
    # =========================================================================
    # 5. Créer movies_collections.csv
    # =========================================================================
    print("\n5. Création de movies_collections.csv...")
    
    collections_rows = []
    for _, row in df.iterrows():
        movie_id = row['id']
        collection = safe_eval_dict(row['belongs_to_collection'])
        if collection:
            collections_rows.append({
                'movie_id': movie_id,
                'collection_id': collection.get('id'),
                'collection_name': collection.get('name')
            })
    
    collections_df = pd.DataFrame(collections_rows)
    collections_df.to_csv(os.path.join(OUTPUT_DIR, "movies_collections.csv"), index=False)
    print(f"  -> {len(collections_df)} relations film-collection exportées")
    
    # =========================================================================
    # Résumé
    # =========================================================================
    print("\n" + "="*60)
    print("NORMALISATION TERMINÉE")
    print("="*60)
    print(f"Fichiers créés dans {OUTPUT_DIR}:")
    print(f"  - movies_normalized.csv  ({len(movies_norm)} lignes)")
    print(f"  - movies_genres.csv      ({len(genres_df)} lignes)")
    print(f"  - movies_companies.csv   ({len(companies_df)} lignes)")
    print(f"  - movies_countries.csv   ({len(countries_df)} lignes)")
    print(f"  - movies_collections.csv ({len(collections_df)} lignes)")
    print("\nVous pouvez maintenant exécuter RMLMapper avec:")
    print("  java -jar rmlmapper.jar -m movies_mapping.rml.ttl -o movies_output.ttl")


if __name__ == "__main__":
    normalize_movies()
