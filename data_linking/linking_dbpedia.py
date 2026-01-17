import time
import requests
import re
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF
from difflib import SequenceMatcher
from datetime import datetime

# --- CONFIGURATION ---
THRESHOLD = 0.9       # On garde le seuil strict de 90%
INPUT_FILE = "../rml/metadata/movies_metadata.ttl"
OUTPUT_TTL = "movie_links_secure.ttl"
LOG_FILE = "execution_trace_v2.txt"

# --- FONCTIONS UTILITAIRES ---

def log_message(file_obj, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    file_obj.write(formatted_msg + "\n")
    file_obj.flush()

def clean_text(text):
    """Supprime les balises HTML <B> et </B> et les espaces en trop"""
    if not text: return ""
    # Enlève les balises XML/HTML
    clean = re.sub('<[^<]+?>', '', text)
    return clean.strip()

def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def lookup_movie_dbpedia(title):
    url = "https://lookup.dbpedia.org/api/search"
    params = {
        "query": title,
        "format": "json",
        "maxResults": 5,  # On regarde les 5 premiers résultats pour trouver le bon
        "typeName": "http://dbpedia.org/ontology/Film"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            docs = data.get("docs", [])
            
            best_match = None
            best_score = 0
            best_label = ""
            
            # On parcourt les candidats pour trouver le meilleur score
            for doc in docs:
                uri = doc["resource"][0]
                
                # Nettoyage du label renvoyé par l'API
                raw_label = doc.get("label", [""])[0]
                if not raw_label:
                    raw_label = uri.split('/')[-1].replace('_', ' ')
                
                cleaned_label = clean_text(raw_label)
                
                # Calcul du score immédiat
                current_score = similar(title, cleaned_label)
                
                # Si ce candidat est meilleur que le précédent, on le garde en mémoire
                if current_score > best_score:
                    best_score = current_score
                    best_match = uri
                    best_label = cleaned_label
            
            # On retourne le meilleur candidat trouvé parmi les 5
            return best_match, best_label, best_score

    except Exception as e:
        return None, f"ERROR: {e}", 0
    
    return None, None, 0

# --- DÉBUT DU TRAITEMENT ---

g = Graph()
MY_NS = Namespace("http://saraaymericradhi.org/movie-ontology#")
links_graph = Graph()
links_graph.bind("owl", OWL)

with open(LOG_FILE, "w", encoding="utf-8") as log:
    log_message(log, f"--- DÉBUT DU TRAITEMENT (CORRIGÉ) ---")
    log_message(log, f"Correction active : Nettoyage des balises HTML + Recherche dans le Top 5")
    
    # Chargement
    try:
        g.parse(INPUT_FILE, format="ttl")
    except Exception as e:
        log_message(log, f"Erreur lecture fichier : {e}")
        exit()

    local_movies = []
    for s, p, o in g.triples((None, MY_NS.hasTitle, None)):
        if (s, RDF.type, MY_NS.Movie) in g:
            local_movies.append((s, str(o)))
            
    log_message(log, f"{len(local_movies)} films à traiter.")
    log_message(log, "------------------------------------------------")

    stats = {"match": 0, "reject": 0, "not_found": 0}
    
    for i, (movie_uri, title) in enumerate(local_movies):
        clean_title = title.strip()
        
        # Appel API qui renvoie maintenant le meilleur score trouvé
        dbpedia_uri, dbpedia_label, score = lookup_movie_dbpedia(clean_title)
        
        if dbpedia_uri and not dbpedia_label.startswith("ERROR"):
            
            if score >= THRESHOLD:
                links_graph.add((movie_uri, OWL.sameAs, URIRef(dbpedia_uri)))
                log_message(log, f"[MATCH  ] Score: {score:.2f} | '{clean_title}' == '{dbpedia_label}'")
                stats["match"] += 1
            else:
                log_message(log, f"[REJET  ] Score: {score:.2f} | '{clean_title}' vs '{dbpedia_label}' (Meilleur résultat insuffisant)")
                stats["reject"] += 1
        else:
            log_message(log, f"[INCONNU] Pas de résultat pertinent pour '{clean_title}'")
            stats["not_found"] += 1
        
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    log_message(log, "------------------------------------------------")
    links_graph.serialize(destination=OUTPUT_TTL, format="turtle")
    log_message(log, f"BILAN : Matches: {stats['match']} | Rejets: {stats['reject']} | Non trouvés: {stats['not_found']}")

print("\nTerminé ! Vérifiez 'execution_trace_v2.txt'")