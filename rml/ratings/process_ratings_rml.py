import csv
import os
import sys

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
# Input CSV is in ../../movies/
input_csv_path = os.path.join(base_dir, '../../movies/ratings_top200.csv')
# Output TTL stays in this folder
output_ttl_path = os.path.join(base_dir, 'ratings_triples.ttl')

PREFIXES = """@prefix : <http://saraaymericradhi.org/movie-ontology#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

"""

def generate_triples(input_file, output_file):
    print(f"Reading {os.path.abspath(input_file)}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            fout.write(PREFIXES)
            
            reader = csv.DictReader(fin)
            count = 0
            
            print("Generating triples (using optimized Python streaming to avoid RMLMapper OOM)...")
            
            for row in reader:
                try:
                    user_id = row['userId']
                    movie_id = row['movieId']
                    rating_val = row['rating']
                    
                    # URIs matching the RML mapping
                    movie_uri = f"http://saraaymericradhi.org/movie/{movie_id}"
                    user_uri = f"http://saraaymericradhi.org/movie/user/{user_id}"
                    rating_uri = f"http://saraaymericradhi.org/movie/rating/{user_id}_{movie_id}"
                    
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
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    generate_triples(input_csv_path, output_ttl_path)
