"""
Flask web server for SPARQL Agent LLM demonstration - Enhanced with Recommendations.
Robust logging + safe JSON handling + no silent failures.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import logging

from sparql_agent import sparql_pipeline, extract_sparql, execute_sparql, PROMPT, LLM as SPARQL_LLM, ONTOLOGY_SCHEMA
from graph_loader import load_graph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import time
import os

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
CORS(app)

# -----------------------------------------------------------------------------
# Logging (THIS IS THE IMPORTANT PART)
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app.logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Request / response logging
# -----------------------------------------------------------------------------

@app.before_request
def log_request():
    app.logger.info(">>> INCOMING REQUEST %s %s", request.method, request.path)
    app.logger.info(">>> Content-Type: %s", request.content_type)
    app.logger.info(">>> Raw body: %s", request.data)

@app.after_request
def log_response(response):
    app.logger.info("<<< RESPONSE STATUS %s", response.status)
    return response

# -----------------------------------------------------------------------------
# Load RDF graph ONCE
# -----------------------------------------------------------------------------

app.logger.info("Loading knowledge graph...")
graph = load_graph()
app.logger.info("Graph loaded (%d triples)", len(graph))

# -----------------------------------------------------------------------------
# Embedding RAG setup (lazy-loaded)
# -----------------------------------------------------------------------------

vector_store = None
embedding_gen = None

def get_embedding_rag():
    """Lazy-load embedding RAG components."""
    global vector_store, embedding_gen
    
    if vector_store is None or embedding_gen is None:
        app.logger.info("Loading embedding RAG components...")
        from embedding_rag import load_vector_store, build_vector_store
        
        # Check if vector store exists
        if os.path.exists("vector_store/entity_index.faiss"):
            vector_store, embedding_gen = load_vector_store()
            app.logger.info("Loaded existing vector store")
        else:
            app.logger.info("Building new vector store (this may take a while)...")
            from embedding_rag import build_vector_store, EmbeddingGenerator
            vector_store = build_vector_store(graph)
            embedding_gen = EmbeddingGenerator()
            app.logger.info("Vector store built successfully")
    
    return vector_store, embedding_gen

# -----------------------------------------------------------------------------
# LLM for answer formatting
# -----------------------------------------------------------------------------

ANSWER_LLM = ChatOpenAI(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
)

ANSWER_TEMPLATE = """
You are a helpful assistant that formats SPARQL query results.

User question:
{question}

Raw SPARQL results:
{results}

Write a short, clear natural language answer.
"""

ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "results"],
    template=ANSWER_TEMPLATE,
)

def format_answer(question: str, raw_results: str) -> str:
    try:
        chain = ANSWER_PROMPT | ANSWER_LLM
        response = chain.invoke({
            "question": question,
            "results": raw_results
        })
        return response.content.strip()
    except Exception as e:
        app.logger.exception("Answer formatting failed")
        return f"Formatting error: {e}\nRaw:\n{raw_results}"

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files (CSS, JS, etc.) from the frontend folder"""
    return send_from_directory("frontend", filename)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "graph_loaded": graph is not None,
        "graph_size": len(graph),
    })

@app.route("/api/ask", methods=["POST"])
def ask_question():
    app.logger.info("===== /api/ask START =====")

    data = request.get_json(silent=True)
    app.logger.info("Parsed JSON: %s", data)

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400

    app.logger.info("QUESTION: %s", question)

    # -------------------------------------------------------------------------
    # STEP 1 – Generate SPARQL
    # -------------------------------------------------------------------------
    app.logger.info("STEP 1: Generating SPARQL query")

    try:
        chain = PROMPT | SPARQL_LLM
        response = chain.invoke({
            "schema": ONTOLOGY_SCHEMA,
            "question": question
        })
        app.logger.info("RAW LLM RESPONSE:\n%s", response.content)
        sparql_query = extract_sparql(response.content)
    except Exception:
        app.logger.exception("SPARQL generation failed")
        return jsonify({"error": "SPARQL generation failed"}), 500

    if not sparql_query:
        return jsonify({"error": "No SPARQL generated"}), 500

    app.logger.info("SPARQL QUERY:\n%s", sparql_query)

    # -------------------------------------------------------------------------
    # STEP 2 – Execute SPARQL
    # -------------------------------------------------------------------------
    app.logger.info("STEP 2: Executing SPARQL")

    try:
        raw_results = sparql_pipeline(question, graph)
    except Exception:
        app.logger.exception("SPARQL execution failed")
        return jsonify({"error": "SPARQL execution failed"}), 500

    app.logger.info("RAW RESULTS:\n%s", raw_results)

    # -------------------------------------------------------------------------
    # STEP 3 – Format answer
    # -------------------------------------------------------------------------
    app.logger.info("STEP 3: Formatting answer")

    answer = format_answer(question, raw_results)

    app.logger.info("FINAL ANSWER:\n%s", answer)
    app.logger.info("===== /api/ask END =====")

    return jsonify({
        "question": question,
        "sparql_query": sparql_query,
        "raw_results": raw_results,
        "answer": answer,
    })

@app.route("/api/ask_embedding", methods=["POST"])
def ask_question_embedding():
    """Embedding-based RAG endpoint."""
    app.logger.info("===== /api/ask_embedding START =====")
    
    data = request.get_json(silent=True)
    app.logger.info("Parsed JSON: %s", data)
    
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400
    
    app.logger.info("QUESTION: %s", question)
    
    # Load embedding RAG components
    try:
        start_time = time.time()
        vs, emb_gen = get_embedding_rag()
        load_time = time.time() - start_time
        app.logger.info("Loaded embedding RAG in %.2fs", load_time)
    except Exception:
        app.logger.exception("Failed to load embedding RAG")
        return jsonify({"error": "Failed to load embedding RAG"}), 500
    
    # Run embedding RAG pipeline
    try:
        start_time = time.time()
        from embedding_rag import embedding_rag_pipeline
        result = embedding_rag_pipeline(question, vs, emb_gen)
        processing_time = time.time() - start_time
        app.logger.info("Processing time: %.2fs", processing_time)
    except Exception:
        app.logger.exception("Embedding RAG failed")
        return jsonify({"error": "Embedding RAG failed"}), 500
    
    app.logger.info("ANSWER: %s", result["answer"])
    app.logger.info("===== /api/ask_embedding END =====")
    
    return jsonify({
        "question": question,
        "answer": result["answer"],
        "retrieved_entities": result["retrieved_entities"],
        "retrieved_triplets": result["retrieved_triplets"],
        "num_entities": result["num_entities"],
        "num_triplets": result["num_triplets"],
        "processing_time": processing_time,
        "approach": "embedding"
    })

@app.route("/api/compare", methods=["POST"])
def compare_approaches():
    """Compare both SPARQL and Embedding approaches on the same question."""
    app.logger.info("===== /api/compare START =====")
    
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400
    
    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400
    
    app.logger.info("QUESTION: %s", question)
    
    results = {}
    
    # --------------------------------------------------
    # SPARQL Approach
    # --------------------------------------------------
    app.logger.info("Running SPARQL approach...")
    try:
        start_time = time.time()
        
        chain = PROMPT | SPARQL_LLM
        response = chain.invoke({
            "schema": ONTOLOGY_SCHEMA,
            "question": question
        })
        sparql_query = extract_sparql(response.content)
        raw_results = sparql_pipeline(question, graph)
        answer_sparql = format_answer(question, raw_results)
        
        sparql_time = time.time() - start_time
        
        results["sparql"] = {
            "answer": answer_sparql,
            "query": sparql_query,
            "raw_results": raw_results,
            "processing_time": sparql_time
        }
        app.logger.info("SPARQL done in %.2fs", sparql_time)
    except Exception as e:
        app.logger.exception("SPARQL approach failed")
        results["sparql"] = {"error": str(e)}
    
    # --------------------------------------------------
    # Embedding Approach
    # --------------------------------------------------
    app.logger.info("Running Embedding approach...")
    try:
        start_time = time.time()
        vs, emb_gen = get_embedding_rag()
        
        from embedding_rag import embedding_rag_pipeline
        result = embedding_rag_pipeline(question, vs, emb_gen)
        
        embedding_time = time.time() - start_time
        
        results["embedding"] = {
            "answer": result["answer"],
            "num_entities": result["num_entities"],
            "num_triplets": result["num_triplets"],
            "processing_time": embedding_time
        }
        app.logger.info("Embedding done in %.2fs", embedding_time)
    except Exception as e:
        app.logger.exception("Embedding approach failed")
        results["embedding"] = {"error": str(e)}
    
    app.logger.info("===== /api/compare END =====")
    
    return jsonify({
        "question": question,
        "results": results
    })

@app.route("/api/sparql", methods=["POST"])
def execute_raw_sparql():
    """Execute raw SPARQL query from frontend."""
    app.logger.info("===== /api/sparql START =====")
    
    data = request.get_json(silent=True)
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data["query"]
    app.logger.info("QUERY: %s", query)
    
    try:
        # execute_sparql returns list of rdflib.query.ResultRow
        rows = execute_sparql(graph, query)
        
        if isinstance(rows, str): # Error message
            return jsonify({"error": rows}), 400
            
        # Convert to JSON-friendly format
        json_results = []
        for row in rows:
            result_dict = {}
            # row is ResultRow, can be accessed like dict but need to iterate variables
            # or use .asdict() if available, but let's be safe with manual extraction
            # rdflib ResultRow behaves like a dict of Variable -> Term
            if hasattr(row, 'asdict'):
                # New rdflib
                d = row.asdict()
                for k, v in d.items():
                    result_dict[str(k)] = {"type": type(v).__name__, "value": str(v)}
            else:
                # Fallback
                for k in row.labels: # labels are the variable names
                    val = row[k]
                    if val is not None:
                        result_dict[k] = {"type": type(val).__name__, "value": str(val)}
            
            json_results.append(result_dict)
            
        app.logger.info("Returned %d rows", len(json_results))
        return jsonify(json_results)
        
    except Exception as e:
        app.logger.exception("SPARQL Endpoint Failed")
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------------------
# NEW: Recommendation System Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/recommendations/users", methods=["GET"])
def get_user_profiles():
    """Get all user profiles from the graph."""
    app.logger.info("===== /api/recommendations/users START =====")
    
    try:
        query = """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        
        SELECT DISTINCT ?user ?name 
            (COUNT(DISTINCT ?rating) as ?ratingsCount)
            (AVG(?ratingValue) as ?avgRating)
        WHERE {
            ?user a :User ;
                  foaf:name ?name .
            
            OPTIONAL {
                ?rating a :Rating ;
                       :ratedBy ?user ;
                       :hasRatingValue ?ratingValue .
            }
        }
        GROUP BY ?user ?name
        ORDER BY DESC(?ratingsCount)
        """
        
        rows = execute_sparql(graph, query)
        
        users = []
        for row in rows:
            user_data = row.asdict() if hasattr(row, 'asdict') else {k: row[k] for k in row.labels if row[k] is not None}
            
            users.append({
                "id": str(user_data.get('user', '')),
                "name": str(user_data.get('name', 'Unknown')),
                "ratingsCount": int(user_data.get('ratingsCount', 0)),
                "avgRating": float(user_data.get('avgRating', 0))
            })
        
        app.logger.info(f"Found {len(users)} users")
        return jsonify(users)
        
    except Exception as e:
        app.logger.exception("Failed to get user profiles")
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommendations/user/<path:user_id>/profile", methods=["GET"])
def get_user_profile_details(user_id):
    """Get detailed profile for a specific user including preferences."""
    app.logger.info(f"===== /api/recommendations/user/{user_id}/profile START =====")
    
    try:
        # Get user ratings
        ratings_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        
        SELECT ?movie ?title ?ratingValue
            (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") as ?genres)
            ?director
        WHERE {{
            ?rating a :Rating ;
                   :ratedBy <{user_id}> ;
                   :ratesMovie ?movie ;
                   :hasRatingValue ?ratingValue .
            
            ?movie :hasTitle ?title .
            
            OPTIONAL {{
                ?movie :hasContent [ :hasGenre [ skos:prefLabel ?genreLabel ] ] .
            }}
            
            OPTIONAL {{
                ?movie :hasProduction [ :hasCrewMember [ :personName ?director ; :job "Director" ] ] .
            }}
        }}
        GROUP BY ?movie ?title ?ratingValue ?director
        ORDER BY DESC(?ratingValue)
        """
        
        ratings_rows = execute_sparql(graph, ratings_query)
        
        ratings = []
        genre_counts = {}
        director_counts = {}
        
        for row in ratings_rows:
            row_data = row.asdict() if hasattr(row, 'asdict') else {k: row[k] for k in row.labels if row[k] is not None}
            
            rating_value = float(row_data.get('ratingValue', 0))
            genres_str = str(row_data.get('genres', ''))
            director = str(row_data.get('director', ''))
            
            ratings.append({
                "movieId": str(row_data.get('movie', '')),
                "title": str(row_data.get('title', '')),
                "rating": rating_value
            })
            
            # Count genres (weighted by rating)
            if genres_str:
                for genre in genres_str.split(', '):
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + rating_value
            
            # Count directors (weighted by rating)
            if director:
                director_counts[director] = director_counts.get(director, 0) + rating_value
        
        # Normalize preferences to 0-1 scale
        max_genre_score = max(genre_counts.values()) if genre_counts else 1
        max_director_score = max(director_counts.values()) if director_counts else 1
        
        preferences = {
            "genres": {k: v/max_genre_score for k, v in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)},
            "directors": {k: v/max_director_score for k, v in sorted(director_counts.items(), key=lambda x: x[1], reverse=True)}
        }
        
        profile = {
            "ratings": ratings,
            "preferences": preferences,
            "stats": {
                "totalRatings": len(ratings),
                "avgRating": sum(r["rating"] for r in ratings) / len(ratings) if ratings else 0
            }
        }
        
        app.logger.info(f"User profile: {len(ratings)} ratings, {len(genre_counts)} genres, {len(director_counts)} directors")
        return jsonify(profile)
        
    except Exception as e:
        app.logger.exception(f"Failed to get user profile for {user_id}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommendations/generate", methods=["POST"])
def generate_recommendations():
    """Generate movie recommendations for a user."""
    app.logger.info("===== /api/recommendations/generate START =====")
    
    data = request.get_json(silent=True)
    if not data or "userId" not in data:
        return jsonify({"error": "Missing 'userId' field"}), 400
    
    user_id = data["userId"]
    algorithm = data.get("algorithm", "hybrid")  # hybrid, semantic, collaborative
    limit = data.get("limit", 12)
    
    app.logger.info(f"Generating recommendations for user {user_id} using {algorithm} algorithm")
    
    try:
        # Get user's rated movies to exclude them
        rated_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        
        SELECT DISTINCT ?movie WHERE {{
            ?rating :ratedBy <{user_id}> ;
                   :ratesMovie ?movie .
        }}
        """
        
        rated_rows = execute_sparql(graph, rated_query)
        rated_movie_ids = [str(row.asdict()['movie'] if hasattr(row, 'asdict') else row['movie']) for row in rated_rows]
        
        # Get candidate movies (not rated by user)
        candidates_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        
        SELECT DISTINCT ?movie ?title ?year ?rating
            (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") as ?genres)
            ?director
            ?runtime
            ?posterColor
        WHERE {{
            ?movie a :Movie ;
                   :hasTitle ?title .
            
            OPTIONAL {{
                ?movie :hasInformation [
                    :hasReleaseDate ?date ;
                    :hasRuntime ?runtime
                ] .
                BIND(YEAR(?date) as ?year)
            }}
            
            OPTIONAL {{
                ?movie :hasResult [ :hasVoteAverage ?rating ] .
            }}
            
            OPTIONAL {{
                ?movie :hasContent [ :hasGenre [ skos:prefLabel ?genreLabel ] ] .
            }}
            
            OPTIONAL {{
                ?movie :hasProduction [ :hasCrewMember [ :personName ?director ; :job "Director" ] ] .
            }}
            
            OPTIONAL {{
                ?movie :hasPosterColor ?posterColor .
            }}
            
            FILTER(?movie NOT IN (<{''>, <'.join(rated_movie_ids)}>))
        }}
        GROUP BY ?movie ?title ?year ?rating ?director ?runtime ?posterColor
        LIMIT 100
        """
        
        candidates_rows = execute_sparql(graph, candidates_query)
        
        # Get user preferences
        user_profile_response = get_user_profile_details(user_id)
        user_profile = user_profile_response.get_json()
        
        # Score each candidate movie
        recommendations = []
        
        for row in candidates_rows:
            row_data = row.asdict() if hasattr(row, 'asdict') else {k: row[k] for k in row.labels if row[k] is not None}
            
            movie = {
                "id": str(row_data.get('movie', '')),
                "title": str(row_data.get('title', '')),
                "year": int(row_data.get('year', 0)) if row_data.get('year') else None,
                "rating": float(row_data.get('rating', 0)) if row_data.get('rating') else 0,
                "genres": str(row_data.get('genres', '')).split(', ') if row_data.get('genres') else [],
                "director": str(row_data.get('director', '')),
                "runtime": int(row_data.get('runtime', 0)) if row_data.get('runtime') else 0,
                "posterColor": str(row_data.get('posterColor', '#333'))
            }
            
            # Calculate similarity score
            score, reasons = calculate_movie_score(movie, user_profile, algorithm)
            
            recommendations.append({
                "movie": movie,
                "score": score,
                "reasons": reasons
            })
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        top_recommendations = recommendations[:limit]
        
        app.logger.info(f"Generated {len(top_recommendations)} recommendations")
        return jsonify({
            "userId": user_id,
            "algorithm": algorithm,
            "recommendations": top_recommendations
        })
        
    except Exception as e:
        app.logger.exception("Failed to generate recommendations")
        return jsonify({"error": str(e)}), 500

def calculate_movie_score(movie, user_profile, algorithm):
    """Calculate similarity score between a movie and user profile."""
    score = 0
    reasons = []
    
    user_prefs = user_profile.get("preferences", {})
    genre_prefs = user_prefs.get("genres", {})
    director_prefs = user_prefs.get("directors", {})
    avg_rating = user_profile.get("stats", {}).get("avgRating", 7.0)
    
    # Semantic similarity (content-based)
    semantic_score = 0
    
    # Genre matching
    if movie["genres"]:
        genre_scores = [genre_prefs.get(g, 0) for g in movie["genres"]]
        if genre_scores:
            genre_match = sum(genre_scores) / len(genre_scores)
            semantic_score += genre_match * 0.5
            
            if genre_match > 0.5:
                top_genres = [g for g in movie["genres"] if genre_prefs.get(g, 0) > 0.5]
                if top_genres:
                    reasons.append(f"Genres similaires: {', '.join(top_genres[:2])}")
    
    # Director matching
    if movie["director"]:
        director_match = director_prefs.get(movie["director"], 0)
        semantic_score += director_match * 0.3
        
        if director_match > 0.5:
            reasons.append(f"Réalisateur favori: {movie['director']}")
    
    # Rating similarity
    if movie["rating"]:
        rating_diff = abs(movie["rating"] - avg_rating)
        rating_score = max(0, 1 - rating_diff / 5)
        semantic_score += rating_score * 0.2
    
    # Collaborative filtering (simplified - based on popularity)
    collaborative_score = movie["rating"] / 10 if movie["rating"] else 0.5
    
    # Combine scores based on algorithm
    if algorithm == "semantic":
        score = semantic_score
    elif algorithm == "collaborative":
        score = collaborative_score
        if movie["rating"] > 8.0:
            reasons.append(f"Film hautement noté ({movie['rating']}/10)")
    else:  # hybrid
        # Weight based on number of user ratings
        ratings_count = user_profile.get("stats", {}).get("totalRatings", 0)
        
        if ratings_count < 5:  # Cold start
            semantic_weight = 0.7
            collaborative_weight = 0.3
        elif ratings_count < 15:  # Active user
            semantic_weight = 0.5
            collaborative_weight = 0.5
        else:  # Expert user
            semantic_weight = 0.6
            collaborative_weight = 0.4
        
        score = semantic_score * semantic_weight + collaborative_score * collaborative_weight
    
    # Normalize score to percentage
    score = min(score * 100, 100)
    
    return score, reasons[:3]  # Return top 3 reasons

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.logger.info("=" * 60)
    app.logger.info("SPARQL Agent Server Starting (Enhanced with Recommendations)")
    app.logger.info("http://localhost:5000")
    app.logger.info("=" * 60)

    # IMPORTANT: debug=True + use_reloader=False for stable logs
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
