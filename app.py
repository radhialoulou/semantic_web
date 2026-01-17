"""
Flask web server for SPARQL Agent LLM demonstration.
Robust logging + safe JSON handling + no silent failures.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import logging

from sparql_agent import sparql_pipeline, extract_sparql, PROMPT, LLM as SPARQL_LLM, ONTOLOGY_SCHEMA
from graph_loader import load_graph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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
# LLM for answer formatting
# -----------------------------------------------------------------------------

ANSWER_LLM = ChatOpenAI(
    model="gpt-oss:20b",
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

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.logger.info("=" * 60)
    app.logger.info("SPARQL Agent Server Starting")
    app.logger.info("http://localhost:5000")
    app.logger.info("=" * 60)

    # IMPORTANT: debug=True + use_reloader=False for stable logs
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
