import os
import rdflib

# Define all file paths
GRAPH_FILES = [
    r"d:\semantic_web\ontology\ontology.ttl",
    r"d:\semantic_web\ontology\genres_skos.ttl",
    r"d:\semantic_web\ontology\keywords_skos.ttl",
    r"d:\semantic_web\unstructered_data_extraction\llm_unstructred.ttl",
    r"d:\semantic_web\rml\metadata\movies_metadata.ttl",
    r"d:\semantic_web\rml\keywords\keyword_triples_rml.ttl",
    r"d:\semantic_web\rml\plots\wiki_movies.ttl",
    r"d:\semantic_web\data_linking\movie_links_secure.ttl",
    r"d:\semantic_web\ontology\suggested_alignments.ttl"
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
