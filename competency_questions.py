import rdflib
import time
import os
from rdflib import Graph
from graph_loader import load_graph


# Defines the Competency Questions and their SPARQL queries
COMPETENCY_QUESTIONS = [
    {
        "id": "CQ1",
        "question": "What are the top 3 genres with the highest average movie rating? (considering only genres with at least 5 movies)",
        "query": """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?genre (AVG(?val) as ?avgRating) (COUNT(?movie) as ?movieCount)
        WHERE {
            ?movie :hasContent/:hasGenre ?genre ;
                   :hasRating ?rating .
            ?rating :hasValue ?val .
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
             ?movie :hasProduction/:hasCrewMember ?crew .
             ?crew :job "Director" ;
                   :personName ?directorName .
             ?movie :hasContent/:hasGenre ?genre .
        }
        GROUP BY ?directorName
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
            {
                SELECT (AVG(?r) AS ?avgRevenue) WHERE {
                    ?m :hasResult/:hasRevenue ?r .
                }
            }
            ?movie :hasResult/:hasRevenue ?revenue ;
                   :hasTitle ?title .
            FILTER (?revenue > ?avgRevenue)
        }
        ORDER BY DESC(?revenue)
        LIMIT 10
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
            
            # STANDARD EXECUTION
            results = g.query(cq['query'])
            
            # Materialize results to list
            results_list = list(results)
            elapsed = time.time() - start_time
            
            # Print column headers
            if results.vars:
                print(f"{' | '.join(results.vars)}")
                print("-" * 30)
            
            row_count = 0
            for row in results_list:
                values = [str(val) for val in row]
                print(" | ".join(values))
                row_count += 1
                
            print(f"\n(Found {row_count} results in {elapsed:.4f} seconds)")

        except Exception as e:
            print(f"Query Error: {e}")


if __name__ == "__main__":
    run_competency_questions()
