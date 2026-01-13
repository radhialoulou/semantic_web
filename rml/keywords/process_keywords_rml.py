import csv
import re
import json
import os
import ast
import subprocess
import sys

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes the script is in 'movies/' and SKOS is in 'ontology/' relative to parent workspace
skos_path = os.path.join(base_dir, '../../ontology/keywords_skos.ttl')
input_csv_path = os.path.join(base_dir, '../../movies/keywords_top200.csv')
flat_csv_path = os.path.join(base_dir, 'keywords_for_rml.csv')
rml_mapping_path = os.path.join(base_dir, 'keywords_rml_mapping.ttl')
output_ttl_path = os.path.join(base_dir, 'keyword_triples_rml.ttl')

BASE_URI_KEYWORD = "http://saraaymericradhi.org/movie-ontology/keywords#"

def get_id_uri_map(skos_file):
    """Parses SKOS file to map keyword IDs to their Concept names."""
    id_map = {}
    if not os.path.exists(skos_file):
        print(f"Error: SKOS file not found at {skos_file}")
        return id_map

    with open(skos_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_subject = None
    notation_re = re.compile(r'skos:notation\s+"(\d+)"\^\^xsd:integer')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract concept name (e.g., :ThreeD a skos:Concept)
        if line.startswith(':') and ' a ' in line:
            parts = line.split()
            if parts[0].startswith(':'):
                 current_subject = parts[0][1:] 
        
        # Extract ID (notation)
        not_match = notation_re.search(line)
        if not_match and current_subject:
            notation_id = int(not_match.group(1))
            id_map[notation_id] = current_subject
            
    return id_map

def generate_flat_csv(input_csv, output_csv, id_map):
    """Reads the raw movies CSV and creates a flat CSV (movieId, keywordUri) for RML."""
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV not found at {input_csv}")
        return 0

    print(f"Reading from {input_csv}...")
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', encoding='utf-8', newline='') as fout:
        
        reader = csv.DictReader(fin)
        fieldnames = ['movieId', 'keywordUri']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        
        count_kept = 0
        
        for row in reader:
            movie_id = row['id']
            raw_keywords = row['keywords']
            keywords_list = []
            try:
                # Robust parsing of JSON-like strings
                if raw_keywords.strip().startswith('['):
                    try:
                        keywords_list = json.loads(raw_keywords.replace("'", '"'))
                    except json.JSONDecodeError:
                        keywords_list = ast.literal_eval(raw_keywords)
            except (ValueError, SyntaxError) as e:
                # print(f"Skipping malformed row {movie_id}: {e}")
                continue

            if not isinstance(keywords_list, list):
                 continue

            for kw in keywords_list:
                kw_id = kw.get('id')
                if kw_id is not None:
                    try:
                        kid = int(kw_id)
                        if kid in id_map:
                            concept_name = id_map[kid]
                            full_uri = f"{BASE_URI_KEYWORD}{concept_name}"
                            writer.writerow({
                                'movieId': movie_id,
                                'keywordUri': full_uri
                            })
                            count_kept += 1
                    except ValueError:
                        pass
        return count_kept

def run_rml_mapper(mapping_file, output_file):
    """Runs the RML Mapper via Docker."""
    # We mount the workspace root to /data
    workspace_root = os.path.dirname(base_dir) 
    
    # Calculate relative paths for Docker container
    rel_mapping = os.path.relpath(mapping_file, workspace_root).replace('\\', '/')
    rel_output = os.path.relpath(output_file, workspace_root).replace('\\', '/')
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace_root}:/data",
        "rmlio/rmlmapper-java",
        "-m", f"/data/{rel_mapping}",
        "-o", f"/data/{rel_output}",
        "-s", "turtle"
    ]
    
    print(f"Running Docker command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stderr:
            print("RML Mapper Log:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error running RML Mapper:")
        print(e.stderr)
        print(e.stdout)

if __name__ == '__main__':
    print("--- Step 1: Extract SKOS mappings ---")
    id_map = get_id_uri_map(skos_path)
    print(f"Found {len(id_map)} concept mappings from {os.path.basename(skos_path)}.")

    print("\n--- Step 2: Generate RML-ready CSV ---")
    kept = generate_flat_csv(input_csv_path, flat_csv_path, id_map)
    print(f"Generated {kept} movie-keyword links in {os.path.basename(flat_csv_path)}")

    print("\n--- Step 3: Execute RML Mapper ---")
    if kept > 0:
        run_rml_mapper(rml_mapping_path, output_ttl_path)
        if os.path.exists(output_ttl_path):
             print(f"\nSuccess! Output saved to: {output_ttl_path}")
    else:
        print("No filtering results. Skipping RML processing.")
