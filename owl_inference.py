"""
OWL Inference Pipeline
======================
Charge les graphes RDF (sauf ratings), applique un raisonneur OWL et les règles
SPARQL CONSTRUCT, puis enregistre chaque graphe enrichi dans un dossier turtle.
"""

import os
import sys
import io
import re
import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.plugins.sparql import prepareQuery
from owlrl import DeductiveClosure, OWLRL_Semantics

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "inferred_graphs")

# Namespace
MOVIE = Namespace("http://saraaymericradhi.org/movie-ontology#")

# Fichiers à charger (sans les ratings)
GRAPH_FILES = {
    "ontology_shapes": os.path.join(BASE_DIR, "ontology", "shapes.ttl"),
    "genres_skos": os.path.join(BASE_DIR, "ontology", "genres_skos.ttl"),
    "keywords_skos": os.path.join(BASE_DIR, "ontology", "keywords_skos.ttl"),
    "suggested_alignments": os.path.join(BASE_DIR, "ontology", "suggested_alignments.ttl"),
    "llm_unstructured": os.path.join(BASE_DIR,"rml", "unstructered_data_extraction", "llm_unstructred.ttl"),
    "movies_metadata": os.path.join(BASE_DIR, "rml", "metadata", "movies_metadata.ttl"),
    "keywords_rml": os.path.join(BASE_DIR, "rml", "keywords", "keyword_triples_rml.ttl"),
    "wiki_plots": os.path.join(BASE_DIR, "rml", "plots", "wiki_movies.ttl"),
    "movie_links": os.path.join(BASE_DIR, "data_linking", "movie_links_secure.ttl"),
    "credits": os.path.join(BASE_DIR, "rml","credits", "credits.ttl"),
}

# Règles SPARQL CONSTRUCT (séparées)
SPARQL_RULES = [
    # 1. Inférer la classe Director
    """
    PREFIX movie: <http://saraaymericradhi.org/movie-ontology#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    CONSTRUCT {
        ?crew rdf:type movie:Director .
    }
    WHERE {
        ?crew rdf:type movie:CrewMember ;
              movie:job "Director" .
    }
    """,
    
    # 2. Inférer la classe Writer
    """
    PREFIX movie: <http://saraaymericradhi.org/movie-ontology#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    CONSTRUCT {
        ?crew rdf:type movie:Writer .
    }
    WHERE {
        ?crew rdf:type movie:CrewMember ;
              movie:job "Writer" .
    }
    """,
    
    # 3. Inférer hasDirector
    """
    PREFIX movie: <http://saraaymericradhi.org/movie-ontology#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    CONSTRUCT {
        ?production movie:hasDirector ?director .
    }
    WHERE {
        ?production rdf:type movie:MovieProduction ;
                    movie:hasCrewMember ?director .
        ?director rdf:type movie:Director .
    }
    """,
    
    # 4. Inférer hasWriter
    """
    PREFIX movie: <http://saraaymericradhi.org/movie-ontology#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    CONSTRUCT {
        ?production movie:hasWriter ?writer .
    }
    WHERE {
        ?production rdf:type movie:MovieProduction ;
                    movie:hasCrewMember ?writer .
        ?writer rdf:type movie:Writer .
    }
    """
]


def load_all_graphs():
    """Charge tous les graphes individuellement."""
    graphs = {}
    for name, path in GRAPH_FILES.items():
        if os.path.exists(path):
            g = Graph()
            g.bind("movie", MOVIE)
            g.bind("owl", OWL)
            g.bind("rdfs", RDFS)
            try:
                g.parse(path, format="turtle")
                print(f"✓ Chargé {name}: {len(g)} triples")
                graphs[name] = g
            except Exception as e:
                print(f"✗ Erreur chargement {name}: {e}")
        else:
            print(f"⚠ Fichier non trouvé: {path}")
    return graphs


def load_combined_graph():
    """Charge tous les graphes dans un seul graphe combiné."""
    combined = Graph()
    combined.bind("movie", MOVIE)
    combined.bind("owl", OWL)
    combined.bind("rdfs", RDFS)
    
    total_before = 0
    for name, path in GRAPH_FILES.items():
        if os.path.exists(path):
            try:
                combined.parse(path, format="turtle")
                total_after = len(combined)
                print(f"✓ Chargé {name}: +{total_after - total_before} triples")
                total_before = total_after
            except Exception as e:
                print(f"✗ Erreur chargement {name}: {e}")
        else:
            print(f"⚠ Fichier non trouvé: {path}")
    
    print(f"\n📊 Total combiné: {len(combined)} triples")
    return combined


def apply_owl_reasoning(graph):
    """Applique le raisonneur OWL-RL sur le graphe."""
    initial_count = len(graph)
    print(f"\n🔮 Application du raisonneur OWL-RL...")
    
    try:
        DeductiveClosure(OWLRL_Semantics).expand(graph)
        inferred_count = len(graph) - initial_count
        print(f"   ✓ OWL-RL: +{inferred_count} triples inférés")
    except Exception as e:
        print(f"   ✗ Erreur OWL-RL: {e}")
    
    return graph


def apply_sparql_rules(graph):
    """Applique les règles SPARQL CONSTRUCT sur le graphe."""
    print(f"\n📜 Application des règles SPARQL CONSTRUCT...")
    
    total_inferred = 0
    for i, rule in enumerate(SPARQL_RULES, 1):
        try:
            result = graph.query(rule)
            new_triples = list(result)
            
            for triple in new_triples:
                graph.add(triple)
            
            print(f"   ✓ Règle {i}: +{len(new_triples)} triples")
            total_inferred += len(new_triples)
        except Exception as e:
            print(f"   ✗ Règle {i} erreur: {e}")
    
    print(f"   📊 Total SPARQL: +{total_inferred} triples")
    return graph


def save_individual_graphs(graphs, combined_graph):
    """Sauvegarde chaque graphe enrichi dans le dossier de sortie."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n💾 Sauvegarde des graphes dans {OUTPUT_DIR}...")
    
    # Sauvegarder le graphe combiné enrichi
    combined_path = os.path.join(OUTPUT_DIR, "combined_inferred.ttl")
    combined_graph.serialize(destination=combined_path, format="turtle")
    print(f"   ✓ combined_inferred.ttl: {len(combined_graph)} triples")
    
    # Sauvegarder les graphes originaux + les inférences pertinentes
    for name, original_graph in graphs.items():
        # Créer un sous-graphe avec les triples originaux + inférences liées
        output_path = os.path.join(OUTPUT_DIR, f"{name}_inferred.ttl")
        original_graph.serialize(destination=output_path, format="turtle")
        print(f"   ✓ {name}_inferred.ttl: {len(original_graph)} triples")


def is_valid_uri(uri):
    """Vérifie si une URI est valide pour la sérialisation."""
    if not isinstance(uri, URIRef):
        return True  # Literals et BNodes sont ok
    uri_str = str(uri)
    # Caractères interdits dans les URIs
    invalid_chars = ['"', ' ', '<', '>', '{', '}', '|', '\\', '^', '`']
    return not any(c in uri_str for c in invalid_chars)


def filter_valid_triples(graph):
    """Filtre les triples avec des URIs invalides."""
    valid_graph = Graph()
    valid_graph.bind("movie", MOVIE)
    valid_graph.bind("owl", OWL)
    valid_graph.bind("rdfs", RDFS)
    
    invalid_count = 0
    for s, p, o in graph:
        if is_valid_uri(s) and is_valid_uri(p) and is_valid_uri(o):
            valid_graph.add((s, p, o))
        else:
            invalid_count += 1
    
    return valid_graph, invalid_count


def extract_inferences_per_graph(combined_graph, original_graphs):
    """Extrait les inférences pertinentes pour chaque graphe original."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n💾 Extraction et sauvegarde des inférences par graphe...")
    
    # Filtrer et sauvegarder le graphe combiné complet
    filtered_combined, invalid_count = filter_valid_triples(combined_graph)
    combined_path = os.path.join(OUTPUT_DIR, "combined_full.ttl")
    filtered_combined.serialize(destination=combined_path, format="turtle")
    print(f"   ✓ combined_full.ttl: {len(filtered_combined)} triples (filtré {invalid_count} URIs invalides)")
    
    # Pour chaque graphe original, extraire les sujets et trouver les inférences
    for name, original_graph in original_graphs.items():
        enriched = Graph()
        enriched.bind("movie", MOVIE)
        enriched.bind("owl", OWL)
        enriched.bind("rdfs", RDFS)
        
        # Copier les triples originaux
        for triple in original_graph:
            enriched.add(triple)
        
        # Ajouter les inférences pour les sujets de ce graphe
        subjects = set(original_graph.subjects())
        for s in subjects:
            for p, o in combined_graph.predicate_objects(s):
                enriched.add((s, p, o))
        
        # Filtrer les URIs invalides
        filtered_enriched, invalid = filter_valid_triples(enriched)
        
        output_path = os.path.join(OUTPUT_DIR, f"{name}_enriched.ttl")
        filtered_enriched.serialize(destination=output_path, format="turtle")
        original_count = len(filter_valid_triples(original_graph)[0])
        added = len(filtered_enriched) - original_count
        print(f"   ✓ {name}_enriched.ttl: {original_count} + {added} = {len(filtered_enriched)} triples")


def main():
    print("=" * 60)
    print("🚀 OWL INFERENCE PIPELINE")
    print("=" * 60)
    
    # 1. Charger tous les graphes
    print("\n📂 Chargement des graphes (sans ratings)...")
    individual_graphs = load_all_graphs()
    
    # 2. Charger le graphe combiné
    print("\n📦 Création du graphe combiné...")
    combined = load_combined_graph()
    
    # 3. Appliquer le raisonneur OWL
    combined = apply_owl_reasoning(combined)
    
    # 4. Appliquer les règles SPARQL
    combined = apply_sparql_rules(combined)
    
    # 5. Extraire et sauvegarder les inférences par graphe
    extract_inferences_per_graph(combined, individual_graphs)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline terminé!")
    print(f"📁 Résultats dans: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
