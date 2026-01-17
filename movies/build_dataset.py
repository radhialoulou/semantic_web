import pandas as pd
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Files (Large datasets)
METADATA_PATH = os.path.join(BASE_DIR, 'movies_metadata.csv')
CREDITS_PATH = os.path.join(BASE_DIR, 'credits.csv')
KEYWORDS_PATH = os.path.join(BASE_DIR, 'keywords.csv')
LINKS_PATH = os.path.join(BASE_DIR, 'links.csv')
RATINGS_PATH = os.path.join(BASE_DIR, 'ratings.csv')
WIKI_PATH = os.path.join(BASE_DIR, 'wiki_movie_plots_deduped.csv')

# Output Files (Filtered/Top200)
METADATA_OUT = os.path.join(BASE_DIR, 'movies_metadata_top200.csv')
CREDITS_OUT = os.path.join(BASE_DIR, 'credits_top200.csv')
KEYWORDS_OUT = os.path.join(BASE_DIR, 'keywords_top200.csv')
LINKS_OUT = os.path.join(BASE_DIR, 'links_top200.csv')
RATINGS_OUT = os.path.join(BASE_DIR, 'ratings_top200.csv')
WIKI_OUT = os.path.join(BASE_DIR, 'wiki_movie_plots_top200.csv')

TOP_N = 200

def normalize_title(title):
    if not isinstance(title, str):
        return ""
    # Lowercase, numeric and letters only
    title = title.lower().strip()
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title

def step_1_top_movies():
    print(f"1. Extracting Top {TOP_N} Movies by vote_count...")
    if not os.path.exists(METADATA_PATH):
        print(f"   Error: {METADATA_PATH} not found.")
        return None, None

    try:
        df = pd.read_csv(METADATA_PATH, low_memory=False)
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
        df = df.dropna(subset=['vote_count'])
        df_sorted = df.sort_values(by='vote_count', ascending=False)
        top_movies = df_sorted.head(TOP_N)
        
        top_movies.to_csv(METADATA_OUT, index=False)
        
        # Get IDs and Titles for next steps
        top_movies['id'] = pd.to_numeric(top_movies['id'], errors='coerce')
        valid_ids = set(top_movies['id'].dropna().astype(int))
        
        # Prepare data for fuzzy matching (title + year)
        top_movies['release_year'] = pd.to_datetime(top_movies['release_date'], errors='coerce').dt.year
        valid_movies_list = top_movies[['id', 'title', 'release_year']].to_dict('records')
        
        print(f"   -> Saved {len(top_movies)} movies.")
        return valid_ids, valid_movies_list
    except Exception as e:
        print(f"   Error: {e}")
        return None, None

def filter_by_id(input_path, output_path, id_col, valid_ids):
    print(f"   Processing {os.path.basename(input_path)}...")
    if not os.path.exists(input_path):
        print(f"   Warning: File not found.")
        return None

    try:
        df = pd.read_csv(input_path, low_memory=False)
        # Convert ID col to numeric for matching
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
        df_filtered = df[df[id_col].isin(valid_ids)]
        
        df_filtered.to_csv(output_path, index=False)
        print(f"   -> Filtered to {len(df_filtered)} rows.")
        return df_filtered
    except Exception as e:
        print(f"   Error processing {os.path.basename(input_path)}: {e}")
        return None

def step_wiki_matching(target_movies):
    print(f"   Processing Wiki Plots (Embeddings Matching)...")
    if not os.path.exists(WIKI_PATH):
        print("   Warning: Wiki file not found.")
        return

    try:
        df_wiki = pd.read_csv(WIKI_PATH)
        
        # Charger le modèle
        print("   -> Loading Embedding Model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Préparer les données
        target_titles = [m['title'] if isinstance(m['title'], str) else "" for m in target_movies]
        wiki_titles = df_wiki['Title'].fillna("").astype(str).tolist()
        
        print("   -> Encoding titles...")
        # Encode
        target_embeddings = model.encode(target_titles, convert_to_tensor=True, show_progress_bar=False)
        wiki_embeddings = model.encode(wiki_titles, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute Similarity
        cosine_scores = util.cos_sim(target_embeddings, wiki_embeddings)
        
        matched_indices = {}  # Changed to dict to store (wiki_idx -> movie_id)
        match_count = 0
        
        print("   -> Finding matches...")
        for i in range(len(target_movies)):
            movie = target_movies[i]
            movie_id = int(movie['id'])  # Use existing TMDb id
            movie_year = movie['release_year']
            
            # Top 5 candidats
            top_results = torch.topk(cosine_scores[i], k=5)
            
            best_idx = None
            
            for score, idx in zip(top_results.values, top_results.indices):
                score = score.item()
                idx = idx.item()
                
                wiki_year = df_wiki.iloc[idx]['Release Year']
                
                # Check Year (+/- 1 an)
                is_year_valid = False
                if pd.notna(movie_year) and pd.notna(wiki_year):
                    if abs(movie_year - wiki_year) <= 1:
                        is_year_valid = True
                        
                if is_year_valid and score > 0.5:
                    best_idx = idx
                    break 
            
            if best_idx is not None:
                matched_indices[best_idx] = movie_id  # Store movie_id with index
                match_count += 1

        # Create final dataframe with movie IDs
        df_final = df_wiki.loc[sorted(list(matched_indices.keys()))].copy()
        df_final.insert(0, 'id', [matched_indices[idx] for idx in sorted(list(matched_indices.keys()))])
        df_final.to_csv(WIKI_OUT, index=False)
        print(f"   -> Matched {len(df_final)} plots (Coverage: {match_count}/{len(target_movies)}).")
        
    except Exception as e:
        print(f"   Error processing wiki plots: {e}")

def main():
    # 1. Main Metadata
    valid_tmdb_ids, valid_movies_list = step_1_top_movies()
    if not valid_tmdb_ids:
        return

    # 2. Files linked by TMDb ID
    print("2. Filtering TMDb ID linked files...")
    filter_by_id(KEYWORDS_PATH, KEYWORDS_OUT, 'id', valid_tmdb_ids)
    filter_by_id(CREDITS_PATH, CREDITS_OUT, 'id', valid_tmdb_ids)
    
    # 3. Links (to get MovieLens IDs)
    print("3. Processing Links...")
    df_links = filter_by_id(LINKS_PATH, LINKS_OUT, 'tmdbId', valid_tmdb_ids)
    
    valid_movielens_ids = set()
    if df_links is not None:
        valid_movielens_ids = set(df_links['movieId'].dropna().astype(int))

    # 4. Ratings (via MovieLens ID)
    if valid_movielens_ids:
        print("4. Processing Ratings...")
        # Note: Ratings file is huge
        filter_by_id(RATINGS_PATH, RATINGS_OUT, 'movieId', valid_movielens_ids)

    # 5. Wiki Plots (via Fuzzy Title Match)
    print("5. Processing Wiki Plots...")
    step_wiki_matching(valid_movies_list)

    print("\nDone! Dataset prepared.")

if __name__ == "__main__":
    main()
