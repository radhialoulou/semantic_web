"""
Embedding-based GraphRAG implementation with hybrid approach.
Combines entity-level and triplet-level embeddings for better retrieval.

Architecture:
- Entity-Level: Dense contextual representations of movies, persons, collections
- Triplet-Level: Atomic facts for precise retrieval
- Dual FAISS indexes for efficient similarity search
- LLM-based answer generation with retrieved context
"""

import os
import json
import numpy as np
import rdflib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import faiss
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ============================================================================
# Configuration
# ============================================================================

VECTOR_STORE_DIR = "vector_store"
ENTITY_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "entity_index.faiss")
TRIPLET_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "triplet_index.faiss")
ENTITY_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "entity_metadata.json")
TRIPLET_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "triplet_metadata.json")

# Multilingual model for French + English
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# LLM for answer generation (same as SPARQL approach)
ANSWER_LLM = ChatOpenAI(
    model="llama3.2:3b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
)

# RDF namespaces
MOVIE_NS = rdflib.Namespace("http://saraaymericradhi.org/movie-ontology#")
GENRES_NS = rdflib.Namespace("http://saraaymericradhi.org/movie-ontology/genres#")

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Entity:
    """Represents a complete entity with rich context."""
    id: str
    type: str  # "entity"
    entity_type: str  # "Movie", "Person", "Collection"
    title: str
    text: str  # Natural language representation
    metadata: Dict[str, Any]

@dataclass
class Triplet:
    """Represents an atomic fact (subject-predicate-object)."""
    id: str
    type: str  # "triplet"
    subject: str
    predicate: str
    object: str
    text: str  # Verbalized triplet
    metadata: Dict[str, Any]

# ============================================================================
# Entity Extractor
# ============================================================================

class EntityExtractor:
    """Extracts entities and triplets from RDF graph."""
    
    def __init__(self, graph: rdflib.Graph):
        self.graph = graph
        self.movie_ns = MOVIE_NS
        self.genres_ns = GENRES_NS
    
    def extract_all(self) -> Tuple[List[Entity], List[Triplet]]:
        """Extract both entities and triplets."""
        print("Extracting entities and triplets from RDF graph...")
        entities = self._extract_entities()
        triplets = self._extract_triplets()
        print(f"Extracted {len(entities)} entities and {len(triplets)} triplets")
        return entities, triplets
    
    def _extract_entities(self) -> List[Entity]:
        """Extract entity-level representations."""
        entities = []
        
        # Extract movies
        movie_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        SELECT DISTINCT ?movie ?title WHERE {{
            ?movie a :Movie ;
                   :hasTitle ?title .
        }}
        """
        
        for row in self.graph.query(movie_query):
            movie_uri = str(row.movie)
            title = str(row.title)
            
            # Gather movie properties
            movie_data = self._gather_movie_data(row.movie, title)
            
            # Convert to natural language text
            text = self._movie_to_text(movie_data)
            
            entity = Entity(
                id=f"entity_movie_{len(entities)}",
                type="entity",
                entity_type="Movie",
                title=title,
                text=text,
                metadata=movie_data
            )
            entities.append(entity)
        
        # Extract persons (directors, actors)
        entities.extend(self._extract_person_entities())
        
        return entities
    
    def _gather_movie_data(self, movie_uri, title: str) -> Dict[str, Any]:
        """Gather all properties for a movie."""
        data = {"uri": str(movie_uri), "title": title}
        
        # Get genres
        genres = []
        for _, _, genre_uri in self.graph.triples((movie_uri, self.movie_ns.hasContent, None)):
            for _, _, genre in self.graph.triples((genre_uri, self.movie_ns.hasGenre, None)):
                genre_name = genre.split('#')[-1] if '#' in str(genre) else str(genre)
                genres.append(genre_name)
        data["genres"] = genres
        
        # Get director
        for _, _, prod in self.graph.triples((movie_uri, self.movie_ns.hasProduction, None)):
            for _, _, director in self.graph.triples((prod, self.movie_ns.hasDirector, None)):
                for _, _, dir_name in self.graph.triples((director, self.movie_ns.personName, None)):
                    data["director"] = str(dir_name)
                    break
        
        # Get plot
        for _, _, content in self.graph.triples((movie_uri, self.movie_ns.hasContent, None)):
            for _, _, plot in self.graph.triples((content, self.movie_ns.hasPlot, None)):
                data["plot"] = str(plot)
                break
        
        # Get budget, revenue, rating
        for _, _, info in self.graph.triples((movie_uri, self.movie_ns.hasInformation, None)):
            for _, _, budget in self.graph.triples((info, self.movie_ns.hasBudget, None)):
                data["budget"] = float(budget)
                break
            for _, _, date in self.graph.triples((info, self.movie_ns.hasReleaseDate, None)):
                data["release_date"] = str(date)
                break
        
        for _, _, result in self.graph.triples((movie_uri, self.movie_ns.hasResult, None)):
            for _, _, rating in self.graph.triples((result, self.movie_ns.hasVoteAverage, None)):
                data["rating"] = float(rating)
                break
            for _, _, revenue in self.graph.triples((result, self.movie_ns.hasRevenue, None)):
                data["revenue"] = float(revenue)
                break
        
        # Get main cast (top 3)
        cast = []
        for _, _, prod in self.graph.triples((movie_uri, self.movie_ns.hasProduction, None)):
            for _, _, cast_member in self.graph.triples((prod, self.movie_ns.hasCastMember, None)):
                for _, _, person in self.graph.triples((cast_member, self.movie_ns.performedBy, None)):
                    for _, _, name in self.graph.triples((person, self.movie_ns.personName, None)):
                        cast.append(str(name))
                        if len(cast) >= 3:
                            break
                if len(cast) >= 3:
                    break
            if len(cast) >= 3:
                break
        data["cast"] = cast
        
        # Get collection
        for _, _, content in self.graph.triples((movie_uri, self.movie_ns.hasContent, None)):
            for _, _, collection in self.graph.triples((content, self.movie_ns.hasCollection, None)):
                for _, _, coll_name in self.graph.triples((collection, self.movie_ns.collectionName, None)):
                    data["collection"] = str(coll_name)
                    break
        
        return data
    
    def _movie_to_text(self, data: Dict[str, Any]) -> str:
        """Convert movie data to natural language text."""
        parts = []
        
        # Title and year
        title = data.get("title", "Unknown")
        if "release_date" in data:
            year = data["release_date"][:4]
            parts.append(f"{title} ({year})")
        else:
            parts.append(title)
        
        # Genres
        if data.get("genres"):
            genres = ", ".join(data["genres"][:3])
            parts.append(f"is a {genres} movie")
        
        # Director
        if "director" in data:
            parts.append(f"directed by {data['director']}")
        
        # Plot (truncated)
        if "plot" in data:
            plot = data["plot"][:150]
            parts.append(f"Plot: {plot}...")
        
        # Stats
        stats = []
        if "budget" in data:
            stats.append(f"Budget: ${data['budget']/1e6:.0f}M")
        if "revenue" in data:
            stats.append(f"Revenue: ${data['revenue']/1e6:.0f}M")
        if "rating" in data:
            stats.append(f"Rating: {data['rating']:.1f}/10")
        if stats:
            parts.append(", ".join(stats))
        
        # Cast
        if data.get("cast"):
            cast = ", ".join(data["cast"][:3])
            parts.append(f"Starring: {cast}")
        
        # Collection
        if "collection" in data:
            parts.append(f"Part of {data['collection']} collection")
        
        return ". ".join(parts) + "."
    
    def _extract_person_entities(self) -> List[Entity]:
        """Extract person entities (directors, actors)."""
        entities = []
        
        # Get all directors
        director_query = f"""
        PREFIX : <http://saraaymericradhi.org/movie-ontology#>
        SELECT DISTINCT ?person ?name WHERE {{
            ?person a :Director ;
                    :personName ?name .
        }}
        """
        
        for row in self.graph.query(director_query):
            name = str(row.name)
            
            # Get movies directed
            movies = []
            for _, _, prod in self.graph.triples((None, self.movie_ns.hasDirector, row.person)):
                for movie, _, _ in self.graph.triples((None, self.movie_ns.hasProduction, prod)):
                    for _, _, title in self.graph.triples((movie, self.movie_ns.hasTitle, None)):
                        movies.append(str(title))
            
            text = f"{name} is a film director"
            if movies:
                text += f" who directed {', '.join(movies[:5])}"
                if len(movies) > 5:
                    text += f" and {len(movies) - 5} more films"
            text += "."
            
            entity = Entity(
                id=f"entity_director_{len(entities)}",
                type="entity",
                entity_type="Director",
                title=name,
                text=text,
                metadata={"name": name, "movies": movies}
            )
            entities.append(entity)
        
        return entities
    
    def _extract_triplets(self) -> List[Triplet]:
        """
        Extract ALL triplets from the graph and verbalize them generically.
        
        This approach automatically expands intermediate nodes and filters uninformative triplets.
        """
        print("  > Extracting and verbalizing all RDF triplets...")
        triplets = []
        triplet_counter = 0
        
        # Cache for labels to avoid repeated lookups
        label_cache = {}
        
        def get_label(resource) -> str:
            """Get human-readable label for a resource."""
            if resource in label_cache:
                return label_cache[resource]
            
            resource_str = str(resource)
            
            # Check for common label properties
            for label_prop in [
                rdflib.RDFS.label,
                self.movie_ns.hasTitle,
                self.movie_ns.personName,
                self.movie_ns.characterName,
                self.movie_ns.collectionName,
                self.movie_ns.awardName,
                self.movie_ns.songTitle
            ]:
                for _, _, label in self.graph.triples((resource, label_prop, None)):
                    label_cache[resource] = str(label)
                    return str(label)
            
            # Fallback: extract fragment or last part of URI
            if '#' in resource_str:
                label = resource_str.split('#')[-1]
            elif '/' in resource_str:
                label = resource_str.split('/')[-1]
            else:
                label = resource_str
            
            # Clean up
            label = label.replace('_', ' ').replace('-', ' ')
            label_cache[resource] = label
            return label
        
        def verbalize_predicate(pred_uri: str) -> str:
            """Convert predicate URI to natural language verb phrase."""
            pred_str = str(pred_uri)
            
            # Extract local name
            if '#' in pred_str:
                pred_name = pred_str.split('#')[-1]
            elif '/' in pred_str:
                pred_name = pred_str.split('/')[-1]
            else:
                pred_name = pred_str
            
            # Common RDF predicates
            if pred_str == str(rdflib.RDF.type):
                return "is a"
            if 'subClassOf' in pred_str:
                return "is a subclass of"
            
            # Special cases for better natural language
            predicate_map = {
                'hasGenre': 'is a',
                'hasTitle': 'is titled',
                'hasPlot': 'has plot',
                'hasBudget': 'has budget of',
                'hasRevenue': 'earned',
                'hasReleaseDate': 'was released on',
                'hasRuntime': 'has runtime of',
                'hasVoteAverage': 'has rating of',
                'personName': 'is named',
                'performedBy': 'performed by',
                'playsCharacter': 'plays',
                'hasDirector': 'was directed by',
                'hasCastMember': 'stars',
                'collectionName': 'belongs to collection',
                'characterName': 'plays character'
            }
            
            if pred_name in predicate_map:
                return predicate_map[pred_name]
            
            # Convert camelCase to spaces
            import re
            pred_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', pred_name)
            pred_name = pred_name.lower()
            
            # Remove common prefixes
            pred_name = pred_name.replace('has ', '').replace('is ', '')
            
            return pred_name
        
        def is_intermediate_node(resource) -> bool:
            """Check if resource is an intermediate node without meaningful label."""
            if isinstance(resource, rdflib.BNode):
                return True
            
            label = get_label(resource)
            # Check if label is generic/meaningless
            generic_labels = ['information', 'content', 'production', 'result', 
                            'cast member', 'character', 'collection']
            return label.lower() in generic_labels
        
        def should_skip_triplet(subj_label: str, pred_label: str, obj_label: str) -> bool:
            """Filter out uninformative or redundant triplets."""
            # Skip if subject and object are the same
            if subj_label.lower() == obj_label.lower():
                return True
            
            # Skip incomplete triplets
            if not obj_label or obj_label.strip() == '':
                return True
            
            # Skip "same as" without actual value
            if 'same as' in pred_label.lower() and not obj_label:
                return True
            
            # Skip generic labels
            if pred_label in ['label', 'has label'] and subj_label.lower() == obj_label.lower():
                return True
            
            return False
        
        def format_value(pred_label: str, value: str) -> str:
            """Format values based on predicate type."""
            # Budget/Revenue formatting
            if 'budget' in pred_label.lower() or 'revenue' in pred_label.lower():
                try:
                    amount = float(value)
                    if amount >= 1_000_000:
                        return f"${amount/1_000_000:.0f}M"
                    elif amount >= 1_000:
                        return f"${amount/1_000:.0f}K"
                    else:
                        return f"${amount:.0f}"
                except:
                    return value
            
            # Runtime formatting
            if 'runtime' in pred_label.lower():
                try:
                    minutes = int(float(value))
                    return f"{minutes} minutes"
                except:
                    return value
            
            # Rating formatting
            if 'rating' in pred_label.lower() or 'vote' in pred_label.lower():
                try:
                    rating = float(value)
                    return f"{rating:.1f}/10"
                except:
                    return value
            
            return value
        
        def expand_intermediate_node(subject, predicate, intermediate_node):
            """Expand intermediate node to get real values."""
            expanded_triplets = []
            
            # Get subject label
            subj_label = get_label(subject)
            
            # Get all properties of the intermediate node
            for _, pred2, obj2 in self.graph.triples((intermediate_node, None, None)):
                # Skip rdf:type
                if pred2 == rdflib.RDF.type:
                    continue
                
                pred2_label = verbalize_predicate(pred2)
                
                # Handle object
                if isinstance(obj2, rdflib.Literal):
                    obj2_label = str(obj2)
                    # Format the value
                    obj2_label = format_value(pred2_label, obj2_label)
                    # Truncate long literals
                    if len(obj2_label) > 100:
                        obj2_label = obj2_label[:100] + "..."
                else:
                    obj2_label = get_label(obj2)
                
                # Skip if uninformative
                if should_skip_triplet(subj_label, pred2_label, obj2_label):
                    continue
                
                # Create verbalization
                text = f"{subj_label} {pred2_label} {obj2_label}"
                
                expanded_triplets.append({
                    'subject': subj_label,
                    'predicate': pred2_label,
                    'object': obj2_label,
                    'text': text,
                    'metadata': {
                        'subject_uri': str(subject),
                        'predicate_uri': str(pred2),
                        'object_uri': str(obj2) if not isinstance(obj2, rdflib.Literal) else None,
                        'object_literal': str(obj2) if isinstance(obj2, rdflib.Literal) else None,
                        'expanded_from': str(intermediate_node)
                    }
                })
            
            return expanded_triplets
        
        # Iterate over ALL triplets in the graph
        seen_triplets = set()  # Avoid duplicates
        
        for subject, predicate, obj in self.graph:
            # Skip blank nodes as subjects
            if isinstance(subject, rdflib.BNode):
                continue
            
            # Skip ontology/schema triples (focus on data)
            pred_str = str(predicate)
            if any(x in pred_str for x in ['subClassOf', 'subPropertyOf', 'domain', 'range', 
                                           'inverseOf', 'disjointWith', 'equivalentClass']):
                continue
            
            # Check if object is an intermediate node
            if not isinstance(obj, rdflib.Literal) and is_intermediate_node(obj):
                # Expand the intermediate node
                expanded = expand_intermediate_node(subject, predicate, obj)
                for exp_triplet in expanded:
                    triplet_key = (exp_triplet['subject'], exp_triplet['predicate'], exp_triplet['object'])
                    if triplet_key not in seen_triplets:
                        seen_triplets.add(triplet_key)
                        
                        triplet = Triplet(
                            id=f"triplet_{triplet_counter}",
                            type="triplet",
                            subject=exp_triplet['subject'],
                            predicate=exp_triplet['predicate'],
                            object=exp_triplet['object'],
                            text=exp_triplet['text'],
                            metadata=exp_triplet['metadata']
                        )
                        triplets.append(triplet)
                        triplet_counter += 1
                continue
            
            # Get readable labels for direct triplets
            subj_label = get_label(subject)
            pred_label = verbalize_predicate(predicate)
            
            # Handle object (can be literal or resource)
            if isinstance(obj, rdflib.Literal):
                obj_label = str(obj)
                # Format the value
                obj_label = format_value(pred_label, obj_label)
                # Truncate long literals
                if len(obj_label) > 100:
                    obj_label = obj_label[:100] + "..."
            else:
                obj_label = get_label(obj)
            
            # Skip if uninformative
            if should_skip_triplet(subj_label, pred_label, obj_label):
                continue
            
            # Create verbalization
            triplet_key = (subj_label, pred_label, obj_label)
            if triplet_key in seen_triplets:
                continue
            seen_triplets.add(triplet_key)
            
            # Construct natural language text
            text = f"{subj_label} {pred_label} {obj_label}"
            
            triplet = Triplet(
                id=f"triplet_{triplet_counter}",
                type="triplet",
                subject=subj_label,
                predicate=pred_label,
                object=obj_label,
                text=text,
                metadata={
                    "subject_uri": str(subject),
                    "predicate_uri": str(predicate),
                    "object_uri": str(obj) if not isinstance(obj, rdflib.Literal) else None,
                    "object_literal": str(obj) if isinstance(obj, rdflib.Literal) else None
                }
            )
            triplets.append(triplet)
            triplet_counter += 1
            
            # Progress indicator
            if triplet_counter % 1000 == 0:
                print(f"    Processed {triplet_counter} triplets...")
        
        print(f"  > Extracted {len(triplets)} unique triplets")
        return triplets

# ============================================================================
# Embedding Generator
# ============================================================================

class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts, 
            convert_to_numpy=True, 
            batch_size=batch_size,
            show_progress_bar=show_progress
        )

# ============================================================================
# Dual Vector Store
# ============================================================================

class DualVectorStore:
    """Manages dual FAISS indexes for entities and triplets."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.entity_index = None
        self.triplet_index = None
        self.entity_metadata = []
        self.triplet_metadata = []
    
    def build_indexes(self, entities: List[Entity], triplets: List[Triplet], 
                     entity_embeddings: np.ndarray, triplet_embeddings: np.ndarray):
        """Build FAISS indexes from embeddings."""
        print("Building FAISS indexes...")
        
        # Entity index
        self.entity_index = faiss.IndexFlatL2(self.embedding_dim)
        self.entity_index.add(entity_embeddings.astype('float32'))
        self.entity_metadata = [asdict(e) for e in entities]
        
        # Triplet index
        self.triplet_index = faiss.IndexFlatL2(self.embedding_dim)
        self.triplet_index.add(triplet_embeddings.astype('float32'))
        self.triplet_metadata = [asdict(t) for t in triplets]
        
        print(f"Entity index: {self.entity_index.ntotal} vectors")
        print(f"Triplet index: {self.triplet_index.ntotal} vectors")
    
    def search_entities(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar entities."""
        if self.entity_index is None:
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.entity_index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.entity_metadata):
                result = self.entity_metadata[idx].copy()
                result['similarity_score'] = float(dist)
                results.append(result)
        
        return results
    
    def search_triplets(self, query_embedding: np.ndarray, k: int = 7) -> List[Dict[str, Any]]:
        """Search for similar triplets."""
        if self.triplet_index is None:
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.triplet_index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.triplet_metadata):
                result = self.triplet_metadata[idx].copy()
                result['similarity_score'] = float(dist)
                results.append(result)
        
        return results
    
    def save(self, directory: str = VECTOR_STORE_DIR):
        """Save indexes and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        
        if self.entity_index:
            faiss.write_index(self.entity_index, ENTITY_INDEX_PATH)
            with open(ENTITY_METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.entity_metadata, f, ensure_ascii=False, indent=2)
        
        if self.triplet_index:
            faiss.write_index(self.triplet_index, TRIPLET_INDEX_PATH)
            with open(TRIPLET_METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.triplet_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Vector store saved to {directory}/")
    
    def load(self, directory: str = VECTOR_STORE_DIR):
        """Load indexes and metadata from disk."""
        if os.path.exists(ENTITY_INDEX_PATH):
            self.entity_index = faiss.read_index(ENTITY_INDEX_PATH)
            with open(ENTITY_METADATA_PATH, 'r', encoding='utf-8') as f:
                self.entity_metadata = json.load(f)
            print(f"Loaded entity index: {self.entity_index.ntotal} vectors")
        
        if os.path.exists(TRIPLET_INDEX_PATH):
            self.triplet_index = faiss.read_index(TRIPLET_INDEX_PATH)
            with open(TRIPLET_METADATA_PATH, 'r', encoding='utf-8') as f:
                self.triplet_metadata = json.load(f)
            print(f"Loaded triplet index: {self.triplet_index.ntotal} vectors")

# ============================================================================
# RAG Pipeline
# ============================================================================

ANSWER_TEMPLATE = """You are a helpful movie knowledge assistant.

Question: {question}

ENTITIES (global context):
{entities}

RELEVANT FACTS:
{triplets}

Answer the question concisely based on the above information. If the information is not available, say so.
"""

ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "entities", "triplets"],
    template=ANSWER_TEMPLATE
)

def embedding_rag_pipeline(question: str, vector_store: DualVectorStore, 
                          embedding_gen: EmbeddingGenerator,
                          top_k_entities: int = 3, top_k_triplets: int = 7) -> Dict[str, Any]:
    """
    Complete embedding-based RAG pipeline.
    
    Args:
        question: Natural language question
        vector_store: Dual vector store with entities and triplets
        embedding_gen: Embedding generator
        top_k_entities: Number of entities to retrieve
        top_k_triplets: Number of triplets to retrieve
    
    Returns:
        Dictionary with answer and retrieval details
    """
    print(f"\n=== Embedding RAG Pipeline ===")
    print(f"Question: {question}")
    
    # Step 1: Generate query embedding
    query_embedding = embedding_gen.generate(question)
    
    # Step 2: Retrieve from both indexes
    entities = vector_store.search_entities(query_embedding, k=top_k_entities)
    triplets = vector_store.search_triplets(query_embedding, k=top_k_triplets)
    
    print(f"Retrieved {len(entities)} entities and {len(triplets)} triplets")
    
    # Step 3: Format context for LLM
    entity_context = ""
    for i, entity in enumerate(entities, 1):
        entity_context += f"{i}. {entity['text']}\n"
    
    triplet_context = ""
    for triplet in triplets:
        triplet_context += f"- {triplet['text']}\n"
    
    # Step 4: Generate answer with LLM
    chain = ANSWER_PROMPT | ANSWER_LLM
    response = chain.invoke({
        "question": question,
        "entities": entity_context,
        "triplets": triplet_context
    })
    
    answer = response.content.strip()
    
    # Safe print with encoding handling
    try:
        print(f"Answer: {answer}")
    except UnicodeEncodeError:
        print(f"Answer: {answer.encode('utf-8', errors='replace').decode('utf-8')}")
    
    return {
        "question": question,
        "answer": answer,
        "retrieved_entities": entities,
        "retrieved_triplets": triplets,
        "num_entities": len(entities),
        "num_triplets": len(triplets)
    }

# ============================================================================
# Main: Build Vector Store
# ============================================================================

def build_vector_store(graph: rdflib.Graph) -> DualVectorStore:
    """Build vector store from RDF graph."""
    
    # Extract entities and triplets
    extractor = EntityExtractor(graph)
    entities, triplets = extractor.extract_all()
    
    if not entities and not triplets:
        raise ValueError("No entities or triplets extracted!")
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator()
    
    print("\nGenerating entity embeddings...")
    entity_texts = [e.text for e in entities]
    entity_embeddings = embedding_gen.generate_batch(entity_texts)
    
    print("\nGenerating triplet embeddings...")
    triplet_texts = [t.text for t in triplets]
    triplet_embeddings = embedding_gen.generate_batch(triplet_texts)
    
    # Build vector store
    vector_store = DualVectorStore(embedding_dim=embedding_gen.model.get_sentence_embedding_dimension())
    vector_store.build_indexes(entities, triplets, entity_embeddings, triplet_embeddings)
    
    # Save to disk
    vector_store.save()
    
    return vector_store

def load_vector_store() -> Tuple[DualVectorStore, EmbeddingGenerator]:
    """Load existing vector store from disk."""
    vector_store = DualVectorStore()
    vector_store.load()
    embedding_gen = EmbeddingGenerator()
    return vector_store, embedding_gen

# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    from graph_loader import load_graph
    
    print("=" * 70)
    print("EMBEDDING-BASED GRAPHRAG - HYBRID APPROACH")
    print("=" * 70)
    
    # Load graph
    graph = load_graph()
    
    # Build or load vector store
    if os.path.exists(ENTITY_INDEX_PATH) and os.path.exists(TRIPLET_INDEX_PATH):
        print("\nLoading existing vector store...")
        vector_store, embedding_gen = load_vector_store()
    else:
        print("\nBuilding new vector store...")
        vector_store = build_vector_store(graph)
        embedding_gen = EmbeddingGenerator()
    
    # Test questions
    test_questions = [
        "Quel est le budget du film Thor?",
        "Qui a réalisé Iron Man 2?",
        "Quels sont les films d'action?",
        "Parle-moi du film The Avengers",
    ]
    
    print("\n" + "=" * 70)
    print("TESTING EMBEDDING RAG PIPELINE")
    print("=" * 70)
    
    for question in test_questions:
        result = embedding_rag_pipeline(question, vector_store, embedding_gen)
        print("\n" + "-" * 70)
    
    print("\n" + "=" * 70)
    print("Vector store created successfully!")
    print(f"Location: {VECTOR_STORE_DIR}/")
    print("=" * 70)
