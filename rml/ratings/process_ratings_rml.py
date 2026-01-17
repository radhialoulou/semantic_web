import csv
import os
import sys

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
# Input CSV is in ../../movies/
input_csv_path = os.path.join(base_dir, '../../movies/ratings_top200.csv')
links_csv_path = os.path.join(base_dir, '../../movies/links_top200.csv')
# Output TTL stays in this folder
output_ttl_path = os.path.join(base_dir, 'ratings_triples.ttl')

PREFIXES = """@prefix : <http://saraaymericradhi.org/movie-ontology#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

"""

def load_links_mapping(links_file):
    """
    Loads mapping from MovieLens ID (movieId) to TMDB ID (tmdbId).
    """
    mapping = {}
    print(f"Reading links from {os.path.abspath(links_file)}...")
    if not os.path.exists(links_file):
        print(f"Error: Links file not found at {links_file}")
        return mapping
        
    with open(links_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ml_id = row['movieId']
            tmdb_id = row['tmdbId']
            # tmdbId might be float-like string "862.0" in some pandas outputs, handle that
            if tmdb_id:
                try:
                    clean_tmdb = str(int(float(tmdb_id)))
                    mapping[ml_id] = clean_tmdb
                except ValueError:
                    continue
    return mapping

def generate_triples(input_file, output_file, id_mapping):
    print(f"Reading ratings from {os.path.abspath(input_file)}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            fout.write(PREFIXES)
            
            reader = csv.DictReader(fin)
            count = 0
            skipped = 0
            
            print("Generating triples (using id mapping to link to TMDB URIs)...")
            
            for row in reader:
                try:
                    user_id = row['userId']
                    ml_movie_id = row['movieId']
                    rating_val = row['rating']
                    
                    # Convert MovieLens ID to TMDB ID
                    if ml_movie_id not in id_mapping:
                        skipped += 1
                        continue
                        
                    tmdb_id = id_mapping[ml_movie_id]
                    
                    # URIs
                    # Movie URI must use TMDB ID
                    movie_uri = f"http://saraaymericradhi.org/movie/{tmdb_id}"
                    # User Entity
                    user_uri = f"http://saraaymericradhi.org/movie/user/{user_id}"
                    # Rating Entity (unique by user and movie)
                    rating_uri = f"http://saraaymericradhi.org/movie/rating/{user_id}_{tmdb_id}"
                    
                    # Triples Block
                    triples = []
                    
                    # Movie -> Rating
                    triples.append(f"<{movie_uri}> :hasRating <{rating_uri}> .")
                    
                    # Rating Attributes
                    triples.append(f"<{rating_uri}> rdf:type :Rating ;")
                    triples.append(f"    :hasValue \"{rating_val}\"^^xsd:float ;")
                    triples.append(f"    :givenBy <{user_uri}> .")
                    
                    # User Attributes
                    triples.append(f"<{user_uri}> rdf:type :User ;")
                    triples.append(f"    :hasUserId \"{user_id}\"^^xsd:integer .")
                    
                    fout.write("\n".join(triples) + "\n")
                    
                    count += 1
                    if count % 200000 == 0:
                        print(f"Processed {count} ratings...", end='\r')
                except KeyError:
                    continue

        print(f"\nDone. Generated triples for {count} ratings.")
        if skipped > 0:
            print(f"Skipped {skipped} ratings where TMDB ID could not be found.")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    mapping = load_links_mapping(links_csv_path)
    if not mapping:
        print("Warning: No links mapping loaded. URIs might be incorrect if relying on mapping.")
        
    generate_triples(input_csv_path, output_ttl_path, mapping)
