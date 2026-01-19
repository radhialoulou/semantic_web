"""
Flask web server for SPARQL Agent LLM demonstration.
Enhanced with Recommendation System - FULLY CORRECTED VERSION.
Adapted for existing data structure with RDFlib Literal fixes.
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
import math
from collections import defaultdict

# -----------------------------------------------------------------------------
# Flask setup
# -----------------------------------------------------------------------------

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
CORS(app)

# -----------------------------------------------------------------------------
# Logging
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
# UTILITY FUNCTION: Safe extraction from RDFlib Literals
# -----------------------------------------------------------------------------

def safe_extract(value, target_type=str):
    """Safely extract value from RDFlib Literal or other types."""
    if value is None:
        return None
    
    # If it's an RDFlib Literal, extract the value
    if hasattr(value, 'value'):
        value = value.value
    
    # Convert to string first, then to target type
    try:
        str_value = str(value)
        if target_type == int:
            return int(float(str_value))  # Convert via float first for safety
        elif target_type == float:
            return float(str_value)
        else:
            return str_value
    except (ValueError, TypeError):
        return None

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
# Standard Routes
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

    try:
        raw_results = sparql_pipeline(question, graph)
    except Exception:
        app.logger.exception("SPARQL execution failed")
        return jsonify({"error": "SPARQL execution failed"}), 500

    app.logger.info("RAW RESULTS:\n%s", raw_results)

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
    
    try:
        start_time = time.time()
        vs, emb_gen = get_embedding_rag()
        load_time = time.time() - start_time
        app.logger.info("Loaded embedding RAG in %.2fs", load_time)
    except Exception:
        app.logger.exception("Failed to load embedding RAG")
        return jsonify({"error": "Failed to load embedding RAG"}), 500
    
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
        rows = execute_sparql(graph, query)
        
        if isinstance(rows, str):
            return jsonify({"error": rows}), 400
            
        json_results = []
        for row in rows:
            result_dict = {}
            if hasattr(row, 'asdict'):
                d = row.asdict()
                for k, v in d.items():
                    result_dict[str(k)] = {"type": type(v).__name__, "value": str(v)}
            else:
                for k in row.labels:
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
# RECOMMENDATION SYSTEM ALGORITHMS
# -----------------------------------------------------------------------------

def jaccard_similarity(set1, set2):
    """Calcule la similarité de Jaccard entre deux ensembles."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    if not vec1 or not vec2:
        return 0.0
    
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1.keys()).union(vec2.keys()))
    
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)

def pearson_correlation(ratings1, ratings2):
    """Calcule la corrélation de Pearson entre deux ensembles de notes."""
    common_movies = set(ratings1.keys()).intersection(set(ratings2.keys()))
    
    if len(common_movies) < 2:
        return 0.0
    
    sum1 = sum(ratings1[m] for m in common_movies)
    sum2 = sum(ratings2[m] for m in common_movies)
    
    sum1_sq = sum(ratings1[m]**2 for m in common_movies)
    sum2_sq = sum(ratings2[m]**2 for m in common_movies)
    
    p_sum = sum(ratings1[m] * ratings2[m] for m in common_movies)
    
    n = len(common_movies)
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_sq - sum1**2 / n) * (sum2_sq - sum2**2 / n))
    
    if den == 0:
        return 0.0
    
    return num / den

def knn_predict_rating(target_user_ratings, all_users_ratings, movie_id, k=10):
    """Prédit la note d'un film en utilisant K-Nearest Neighbors."""
    similarities = []
    
    for user_id, user_ratings in all_users_ratings.items():
        if movie_id in user_ratings:
            sim = pearson_correlation(target_user_ratings, user_ratings)
            if sim > 0:
                similarities.append((sim, user_ratings[movie_id]))
    
    if not similarities:
        return 0.0
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_k = similarities[:k]
    
    weighted_sum = sum(sim * rating for sim, rating in top_k)
    sum_weights = sum(sim for sim, rating in top_k)
    
    return weighted_sum / sum_weights if sum_weights > 0 else 0.0

# -----------------------------------------------------------------------------
# RECOMMENDATION SYSTEM ENDPOINTS
# -----------------------------------------------------------------------------

@app.route("/api/recommendations/users", methods=["GET"])
def get_user_profiles():
    """Get all user profiles from the graph - OPTIMIZED VERSION."""
    app.logger.info("===== /api/recommendations/users START =====")
    
    try:
        users_query = """
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        
        SELECT DISTINCT ?user ?userId WHERE {
            ?user a :User ;
                  :hasUserId ?userId .
        }
        ORDER BY ?userId
        LIMIT 50
        """
        
        start = time.time()
        rows = execute_sparql(graph, users_query)
        query_time = time.time() - start
        app.logger.info(f"Users query took {query_time:.2f}s")
        
        users = []
        for row in rows:
            user_data = row.asdict() if hasattr(row, 'asdict') else {k: row[k] for k in row.labels if row[k] is not None}
            
            user_uri = str(user_data.get('user', ''))
            user_id = str(user_data.get('userId', ''))
            
            count_query = f"""
            PREFIX : <http://saraaymericradhi.org/movie-ontology#>
            
            SELECT (COUNT(?rating) as ?count) (AVG(?value) as ?avg) WHERE {{
                ?rating a :Rating ;
                       :givenBy <{user_uri}> ;
                       :hasValue ?value .
            }}
            """
            
            count_rows = execute_sparql(graph, count_query)
            count_data = count_rows[0].asdict() if count_rows else {}
            
            ratings_count = safe_extract(count_data.get('count'), int) or 0
            avg_rating = safe_extract(count_data.get('avg'), float) or 0
            
            users.append({
                "id": user_uri,
                "name": f"User {user_id}",
                "ratingsCount": ratings_count,
                "avgRating": round(avg_rating, 1)
            })
        
        users.sort(key=lambda x: x['ratingsCount'], reverse=True)
        
        total_time = time.time() - start
        app.logger.info(f"Found {len(users)} users in {total_time:.2f}s")
        return jsonify(users)
        
    except Exception as e:
        app.logger.exception("Failed to get user profiles")
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommendations/user/<path:user_id>/profile", methods=["GET"])
def get_user_profile_details(user_id):
    """Get detailed profile for a specific user."""
    app.logger.info(f"===== /api/recommendations/user/{user_id}/profile START =====")
    
    try:
        ratings_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        
        SELECT ?movie ?title ?ratingValue
            (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") as ?genres)
            ?director
        WHERE {{
            ?rating a :Rating ;
                   :givenBy <{user_id}> ;
                   :hasValue ?ratingValue .
            
            ?movie :hasRating ?rating ;
                   :hasTitle ?title .
            
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
            
            rating_value = safe_extract(row_data.get('ratingValue'), float) or 0
            genres_str = str(row_data.get('genres', ''))
            director = str(row_data.get('director', ''))
            
            ratings.append({
                "movieId": str(row_data.get('movie', '')),
                "title": str(row_data.get('title', '')),
                "rating": rating_value
            })
            
            if genres_str:
                for genre in genres_str.split(', '):
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + rating_value
            
            if director:
                director_counts[director] = director_counts.get(director, 0) + rating_value
        
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
    algorithm = data.get("algorithm", "hybrid")
    limit = data.get("limit", 12)
    
    app.logger.info(f"Generating recommendations for user {user_id} using {algorithm} algorithm")
    
    try:
        rated_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        
        SELECT DISTINCT ?movie WHERE {{
            ?rating :givenBy <{user_id}> .
            ?movie :hasRating ?rating .
        }}
        """
        
        rated_rows = execute_sparql(graph, rated_query)
        rated_movie_ids = []
        for row in rated_rows:
            movie_uri = row.asdict()['movie'] if hasattr(row, 'asdict') else row['movie']
            rated_movie_ids.append(str(movie_uri))
        
        if rated_movie_ids:
            rated_uris = '>, <'.join(rated_movie_ids)
            filter_clause = f"FILTER(?movie NOT IN (<{rated_uris}>))"
        else:
            filter_clause = ""
        
        candidates_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        
        SELECT DISTINCT ?movie ?title ?year ?rating
            (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") as ?genres)
            ?director
            ?runtime
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
            
            {filter_clause}
        }}
        GROUP BY ?movie ?title ?year ?rating ?director ?runtime
        LIMIT 100
        """
        
        candidates_rows = execute_sparql(graph, candidates_query)
        
        user_profile_response = get_user_profile_details(user_id)
        user_profile = user_profile_response.get_json()
        
        all_users_ratings = get_all_users_ratings() if algorithm in ["knn", "pearson", "hybrid"] else {}
        
        recommendations = []
        
        for row in candidates_rows:
            row_data = row.asdict() if hasattr(row, 'asdict') else {k: row[k] for k in row.labels if row[k] is not None}
            
            movie = {
                "id": str(row_data.get('movie', '')),
                "title": str(row_data.get('title', '')),
                "year": safe_extract(row_data.get('year'), int),
                "rating": safe_extract(row_data.get('rating'), float) or 0,
                "genres": str(row_data.get('genres', '')).split(', ') if row_data.get('genres') else [],
                "director": str(row_data.get('director', '')),
                "runtime": safe_extract(row_data.get('runtime'), int) or 0,
                "posterColor": "#333"
            }
            
            movie["genres"] = [g for g in movie["genres"] if g.strip()]
            
            score, reasons = calculate_movie_score(movie, user_profile, algorithm, all_users_ratings, user_id)
            
            recommendations.append({
                "movie": movie,
                "score": score,
                "reasons": reasons
            })
        
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

def get_all_users_ratings():
    """Récupère toutes les notes de tous les utilisateurs."""
    query = """
    PREFIX : <http://saraaymericradhi.org/movie-ontology#>
    
    SELECT ?user ?movie ?rating WHERE {
        ?ratingNode a :Rating ;
                   :givenBy ?user ;
                   :hasValue ?rating .
        ?movie :hasRating ?ratingNode .
    }
    """
    
    rows = execute_sparql(graph, query)
    all_ratings = defaultdict(dict)
    
    for row in rows:
        row_data = row.asdict() if hasattr(row, 'asdict') else {k: row[k] for k in row.labels if row[k] is not None}
        user = str(row_data.get('user', ''))
        movie = str(row_data.get('movie', ''))
        rating = safe_extract(row_data.get('rating'), float) or 0
        all_ratings[user][movie] = rating
    
    return dict(all_ratings)

def calculate_movie_score(movie, user_profile, algorithm, all_users_ratings, user_id):
    """Calculate similarity score between a movie and user profile."""
    score = 0
    reasons = []
    
    user_prefs = user_profile.get("preferences", {})
    genre_prefs = user_prefs.get("genres", {})
    director_prefs = user_prefs.get("directors", {})
    avg_rating = user_profile.get("stats", {}).get("avgRating", 7.0)
    
    user_ratings = {r["movieId"]: r["rating"] for r in user_profile.get("ratings", [])}
    user_genres = set(genre_prefs.keys())
    movie_genres = set(movie["genres"])
    
    if algorithm == "jaccard":
        score = jaccard_similarity(user_genres, movie_genres) * 100
        if score > 50:
            common_genres = user_genres.intersection(movie_genres)
            reasons.append(f"Genres en commun: {', '.join(list(common_genres)[:2])}")
    
    elif algorithm == "cosine":
        user_vector = genre_prefs
        movie_vector = {g: 1.0 for g in movie["genres"]}
        score = cosine_similarity(user_vector, movie_vector) * 100
        if score > 50:
            reasons.append(f"Similarité vectorielle des genres élevée")
    
    elif algorithm == "pearson":
        if all_users_ratings and user_id in all_users_ratings:
            other_users = {uid: ratings for uid, ratings in all_users_ratings.items() if uid != user_id}
            correlations = []
            for other_id, other_ratings in other_users.items():
                if movie["id"] in other_ratings:
                    corr = pearson_correlation(user_ratings, other_ratings)
                    if corr > 0:
                        correlations.append((corr, other_ratings[movie["id"]]))
            
            if correlations:
                correlations.sort(reverse=True)
                top_corr = correlations[:5]
                weighted_sum = sum(corr * rating for corr, rating in top_corr)
                sum_weights = sum(corr for corr, rating in top_corr)
                predicted_rating = weighted_sum / sum_weights if sum_weights > 0 else 5.0
                score = (predicted_rating / 10) * 100
                reasons.append(f"Note prédite: {predicted_rating:.1f}/10")
        else:
            score = (movie["rating"] / 10) * 100 if movie["rating"] else 50
    
    elif algorithm == "knn":
        if all_users_ratings and user_id in all_users_ratings:
            predicted_rating = knn_predict_rating(user_ratings, all_users_ratings, movie["id"], k=10)
            score = (predicted_rating / 10) * 100 if predicted_rating > 0 else (movie["rating"] / 10) * 100
            if predicted_rating > 0:
                reasons.append(f"KNN note prédite: {predicted_rating:.1f}/10")
        else:
            score = (movie["rating"] / 10) * 100 if movie["rating"] else 50
    
    elif algorithm == "hybrid":
        jaccard_score = jaccard_similarity(user_genres, movie_genres)
        
        user_vector = genre_prefs
        movie_vector = {g: 1.0 for g in movie["genres"]}
        cosine_score = cosine_similarity(user_vector, movie_vector)
        
        knn_score = 0
        if all_users_ratings and user_id in all_users_ratings:
            predicted_rating = knn_predict_rating(user_ratings, all_users_ratings, movie["id"], k=10)
            knn_score = predicted_rating / 10 if predicted_rating > 0 else 0
        
        ratings_count = user_profile.get("stats", {}).get("totalRatings", 0)
        
        if ratings_count < 5:
            w_content = 0.7
            w_collab = 0.3
        elif ratings_count < 15:
            w_content = 0.5
            w_collab = 0.5
        else:
            w_content = 0.4
            w_collab = 0.6
        
        content_score = (jaccard_score * 0.5 + cosine_score * 0.5)
        collab_score = knn_score if knn_score > 0 else (movie["rating"] / 10)
        
        score = (content_score * w_content + collab_score * w_collab) * 100
        
        if jaccard_score > 0.5:
            reasons.append(f"Genres similaires (Jaccard: {jaccard_score:.2f})")
        if knn_score > 7:
            reasons.append(f"Fortement recommandé par KNN ({knn_score:.1f}/10)")
    
    else:
        score = (movie["rating"] / 10) * 100 if movie["rating"] else 50
        reasons.append("Algorithme par défaut")
    
    score = min(score, 100)
    
    return score, reasons[:3]


if __name__ == "__main__":
    app.logger.info("=" * 60)
    app.logger.info("SPARQL Agent Server Starting (with Recommendations)")
    app.logger.info("http://localhost:5000")
    app.logger.info("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)