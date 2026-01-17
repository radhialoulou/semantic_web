
import rdflib
import time
import os
import urllib.request
from rdflib import Graph
from graph_loader import load_graph
try:
    from SPARQLWrapper import SPARQLWrapper, JSON
except ImportError:
    SPARQLWrapper = None

# MONKEY PATCH: Intercept Request creation to force headers for DBpedia compatibility
# rdflib's native SERVICE clause often sends headers rejected by DBpedia (HTTP 406).
# This patch forces a clean Accept header and a valid User-Agent.
original_init = urllib.request.Request.__init__

def new_init(self, url, data=None, headers={}, origin_req_host=None, unverifiable=False, method=None):
    # Remove existing headers to avoid duplicates/conflicts
    if 'accept' in headers: del headers['accept']
    if 'Accept' in headers: del headers['Accept']
    if 'user-agent' in headers: del headers['user-agent']
    if 'User-Agent' in headers: del headers['User-Agent']
    
    # DBpedia prefers these specific headers
    headers['Accept'] = 'application/sparql-results+json, application/xml' 
    headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    
    original_init(self, url, data, headers, origin_req_host, unverifiable, method)

urllib.request.Request.__init__ = new_init


# Defines the Competency Questions and their SPARQL queries
COMPETENCY_QUESTIONS = [
    {
        "id": "CQ1",
        "question": "What are the top 3 genres with the highest average movie rating? (considering only genres with at least 5 movies)",

        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX genres: <http://saraaymericradhi.org/movie-ontology/genres#>

        SELECT ?genre (AVG(?rating) as ?avgRating) (COUNT(?movie) as ?movieCount)
        WHERE {
            ?movie :hasContent ?content .
            ?content :hasGenre ?genre .
            
            ?movie :hasResult ?result .
            ?result :hasVoteAverage ?rating .
        }
        GROUP BY ?genre
        HAVING (COUNT(?movie) >= 5)
        ORDER BY DESC(?avgRating)

        LIMIT 3
        """
    },
    {
        "id": "CQ2",
        "question": "Show directors and the genres of movies they directed.",
        "description": "Finds directors via 'job' property on CrewMember if explicit 'hasDirector' is missing.",
        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        
        SELECT ?directorName (GROUP_CONCAT(DISTINCT ?genre; separator=", ") AS ?genres)
        WHERE {
            # Find directors: Look for CrewMember with job 'Director'
            ?movie :hasProduction/:hasCrewMember ?director .
            ?director :job "Director" ;
                      :personName ?directorName .
            
            # Find genres
            ?movie :hasContent/:hasGenre ?genre .
        }
        GROUP BY ?directorName
        HAVING (COUNT(DISTINCT ?genre) >= 1)
        ORDER BY ?directorName
        LIMIT 10
        """
    },
    {
        "id": "CQ3",
        "question": "Find the movies that have a higher revenue than the average revenue of all movies.",
        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?title ?revenue
        WHERE {
            ?movie :hasTitle ?title ;
                   :hasResult ?result .
            ?result :hasRevenue ?revenue .
            
            {
                SELECT (AVG(?r) as ?avgRevenue) WHERE {
                    ?m :hasResult ?res .
                    ?res :hasRevenue ?r .
                }
            }
            FILTER (?revenue > ?avgRevenue)
        }
        ORDER BY DESC(?revenue)
        LIMIT 10
        """
    },
    {
        "id": "CQ4_FEDERATED",
        "question": "For movies in our graph linked to DBpedia, find their Distributor (missing locally).",
        "description": "This query uses federation to fetch the 'distributor' property from DBpedia, which is not present in our local graph.",

        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?title ?dbpediaUri ?distributorName
        WHERE {
            # Local part
            ?movie :hasTitle ?title .
            ?movie owl:sameAs ?dbpediaUri .
            
            # Federated part
            SERVICE <https://dbpedia.org/sparql> { 
                OPTIONAL {
                    ?dbpediaUri dbo:distributor ?dist .
                    ?dist rdfs:label ?distributorName .
                    FILTER (LANG(?distributorName) = "en")
                }
            }
        }
        LIMIT 5
        """
    },
    {
        "id": "CQ5",
        "question": "What is the favorite genre of the user who gives the highest average ratings?",
        "description": "Complex Nested Query: 1. Find the most generous user (highest avg rating). 2. Find their most watched genre.",
        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?userId ?featuredGenre (COUNT(?movie) as ?genreCount) (SAMPLE(?avgUserRating) as ?userAvgRating) (GROUP_CONCAT(DISTINCT ?movieTitle; separator=", ") AS ?moviesInGenre)
        WHERE {
            {
                # Subquery: Find the user with the MAX average rating
                SELECT ?user (AVG(?score) as ?avgUserRating) 
                WHERE {
                    ?rating :givenBy ?user ;
                            :hasValue ?score .
                }
                GROUP BY ?user
                ORDER BY DESC(?avgUserRating)
                LIMIT 1
            }
            
            # Main Query: Find movies rated by this user and their genres
            ?ratingLink :givenBy ?user ;
                        :hasValue ?val . # Just to ensure connection
            
            # Inverse property to go from Rating to Movie
            ?movie :hasRating ?ratingLink ;
                   :hasContent/:hasGenre ?featuredGenre ;
                   :hasTitle ?movieTitle .
                   
            ?user :hasUserId ?userId .
        }
        GROUP BY ?userId ?featuredGenre
        ORDER BY DESC(?genreCount)
        LIMIT 1
        """
    },
    {
        "id": "CQ7",
        "question": "Who is the actor that appears most frequently in the most popular genre?",
        "description": "Complex Nested Query: 1. Find the most popular genre (most movies). 2. Find the actor with most appearances in that genre.",
        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?genreName ?actorName (COUNT(?movie) as ?appearanceCount)
        WHERE {
            {
                # Subquery: Find the most popular genre
                SELECT ?genre WHERE {
                    ?m :hasContent/:hasGenre ?genre .
                }
                GROUP BY ?genre
                ORDER BY DESC(COUNT(?m))
                LIMIT 1
            }
            
            # Main query: Find movies in this genre and their actors
            ?movie :hasContent/:hasGenre ?genre .
            ?movie :hasProduction/:hasCastMember ?actor .
            ?actor :personName ?actorName .
            
            # Optional: Get genre name for display if available, else use URI
            BIND(?genre as ?genreName)
        }
        GROUP BY ?genreName ?actorName
        ORDER BY DESC(?appearanceCount)
        LIMIT 1
        """
    }
]

def run_competency_questions():

    print("Loading Knowledge Graph...")
    g = load_graph()
    
    # Check owl:sameAs links
    print("\n--- Verifying owl:sameAs Links ---")
    q_link = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?s ?o WHERE { ?s owl:sameAs ?o . }
    """
    for row in g.query(q_link):
        print(f"Link: {row}")

    print("\n" + "="*80)
    print("EXECUTING COMPETENCY QUESTIONS")
    print("="*80)

    for cq in COMPETENCY_QUESTIONS:
        print(f"\n[{cq['id']}] {cq['question']}")
        if "description" in cq:
            print(f"Note: {cq['description']}")
        print("-" * 60)
        

        try:
            start_time = time.time()
            
            # DYNAMIC OPTIMIZATION FOR CQ4 (Federated Query)
            # rdflib has trouble with open federated joins. We optimize by injecting local URIs via VALUES.
            query_to_run = cq['query']
            if cq['id'] == 'CQ4_FEDERATED':
                print("Optimizing Federated Query with VALUES injection...")
                # 1. Fetch local URIs
                q_local_uris = """
                PREFIX : <http://saraaymericradhi.org/movie-ontology#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                SELECT ?dbpediaUri WHERE { ?movie owl:sameAs ?dbpediaUri . } LIMIT 10
                """
                local_uris = [str(row[0]) for row in g.query(q_local_uris)]
                
                if local_uris:
                    # 2. Inject into VALUES clause
                    # Ensure URIs are wrapped in angle brackets
                    values_content = " ".join([f"<{uri}>" for uri in local_uris])
                    
                    # 3. Rewrite query to include VALUES INSIDE the SERVICE block.
                    # This sends the list of URIs to DBpedia, which is much more efficient and bypasses rdflib join issues.
                    
                    # Split at SERVICE block start
                    service_start_marker = "SERVICE <https://dbpedia.org/sparql> {"
                    parts = cq['query'].split(service_start_marker, 1)
                    
                    if len(parts) == 2:
                        # Insert VALUES clause right after the opening brace of SERVICE
                        query_to_run = f"{parts[0]} {service_start_marker} VALUES ?dbpediaUri {{ {values_content} }} {parts[1]}"
                        print(f"Computed Optimized Query (VALUES inside SERVICE, {len(local_uris)} URIs)")
                else:
                    print("No local owl:sameAs links found to federate.")
            
            results = g.query(query_to_run)
            
            # Materialize results to list to ensure execution happens here
            results_list = list(results)
            elapsed = time.time() - start_time
            
            # Print column headers
            if results.vars:
                print(f"{' | '.join(results.vars)}")
                print("-" * 30)
            
            row_count = 0
            for row in results_list:
                # Format each row
                values = [str(val) for val in row]
                print(" | ".join(values))
                row_count += 1
                
            print(f"\n(Found {row_count} results in {elapsed:.4f} seconds)")
            
            # Application-side Federation Fallback
            # Validates that if native engine fails (common with SERVICE), we handle it robustly.
            if row_count == 0 and "SERVICE" in cq['query']:
                 print("Native SERVICE execution returned 0 results (Engine limitation).")
                 print("Switching to Robust Application-side Federation (SPARQLWrapper)...")
                 
                 # Manual Fallback Logic
                 try:
                    # 1. Get the URI from local graph
                    q_local = """
                    PREFIX : <http://saraaymericradhi.org/movie-ontology#>
                    PREFIX owl: <http://www.w3.org/2002/07/owl#>
                    SELECT ?title ?dbpediaUri WHERE {
                        ?movie :hasTitle ?title .
                        ?movie owl:sameAs ?dbpediaUri .
                    } LIMIT 5
                    """
                    local_res = list(g.query(q_local))
                    
                    if local_res:
                        print(f"Federating over {len(local_res)} local movies...")
                        if SPARQLWrapper:
                            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
                            
                            print(f"{'title':<30} | {'dbpediaUri':<50} | {'Distributor'}")
                            print("-" * 100)
                            
                            for title, dbpedia_uri in local_res:
                                # Prioritize Label -> Abstract -> Comment
                                # Query matching the Relaxed CQ4 structure
                                q_remote = f"""
                                PREFIX dbo: <http://dbpedia.org/ontology/>
                                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                SELECT ?distributorName WHERE {{
                                    OPTIONAL {{ 
                                        <{dbpedia_uri}> dbo:distributor ?dist .
                                        ?dist rdfs:label ?distributorName .
                                        FILTER (LANG(?distributorName) = "en")
                                    }}
                                }} LIMIT 1
                                """
                                sparql.setQuery(q_remote)
                                sparql.setReturnFormat(JSON)
                                try:
                                    rem_results = sparql.query().convert()
                                    if rem_results["results"]["bindings"]:
                                        binding = rem_results["results"]["bindings"][0]
                                        # Pick best available description
                                        desc = "N/A"
                                        if "distributorName" in binding: 
                                            desc = binding["distributorName"]["value"]
                                        
                                        # Truncate for display
                                        desc_preview = (desc[:75] + '..') if len(desc) > 75 else desc
                                        print(f"{str(title):<30} | {str(dbpedia_uri):<50} | {desc_preview}")
                                    else:
                                        print(f"{str(title):<30} | {str(dbpedia_uri):<50} | [N/A]")
                                except Exception as e:
                                    print(f"Error querying DBpedia for {title}: {e}")
                        else:
                            print("SPARQLWrapper not installed, cannot perform fallback.")
                    else:
                        print("No local movies with owl:sameAs links found.")
                        
                 except Exception as ex:
                     print(f"Fallback failed: {ex}")

        except Exception as e:
            print(f"Query Error: {e}")


if __name__ == "__main__":
    run_competency_questions()
