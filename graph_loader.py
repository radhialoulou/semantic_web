import os
import rdflib

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all file paths relative to the project root
GRAPH_FILES = [
    os.path.join(BASE_DIR, "ontology", "ontology.ttl"),
    os.path.join(BASE_DIR, "ontology", "genres_skos.ttl"),
    os.path.join(BASE_DIR, "ontology", "keywords_skos.ttl"),
    os.path.join(BASE_DIR, "unstructered_data_extraction", "llm_unstructred.ttl"),
    os.path.join(BASE_DIR, "rml", "metadata", "movies_metadata.ttl"),
    os.path.join(BASE_DIR, "rml", "keywords", "keyword_triples_rml.ttl"),
    os.path.join(BASE_DIR, "rml", "plots", "wiki_movies.ttl"),
    os.path.join(BASE_DIR, "data_linking", "movie_links_secure.ttl"),
    os.path.join(BASE_DIR, "ontology", "suggested_alignments.ttl"),
    os.path.join(BASE_DIR, "credits", "credits.ttl"),
    os.path.join(BASE_DIR, "rml", "ratings", "ratings_sample.ttl")
]

def load_graph():
    """Parses all defined TTL files into a single rdflib Graph."""
    g = rdflib.Graph()
    print("Loading Graph data...")
    for file_path in GRAPH_FILES:
        if os.path.exists(file_path):
            try:
                print(f"Parsing {file_path}...")
                g.parse(file_path, format="ttl")
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
        else:
            print(f"Warning: File not found {file_path}")
            
    print(f"Graph loaded with {len(g)} triples.")
    return g

if __name__ == "__main__":
    g = load_graph()
