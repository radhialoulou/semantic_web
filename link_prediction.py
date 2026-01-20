"""
Link Prediction using TransE with PyKEEN.
"""

import logging
from typing import List, Tuple
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
import torch
import pickle
import os

logger = logging.getLogger(__name__)


class MovieLinkPredictor:
    """Link prediction model for movie recommendations using TransE."""
    
    def __init__(self, model_path="models/transe_model.pkl"):
        self.model = None
        self.entity_to_id = None
        self.relation_to_id = None
        self.id_to_entity = None
        self.model_path = model_path
        self.trained = False
        self.embedding_dim = 128  # Stocker ici
    
    def extract_triples_from_graph(self, graph) -> List[Tuple[str, str, str]]:
        """Extract all triples from RDF graph."""
        logger.info("Extracting triples from RDF graph...")
        
        triples = []
        for subj, pred, obj in graph:
            s = str(subj).split('#')[-1].split('/')[-1]
            p = str(pred).split('#')[-1].split('/')[-1]
            o = str(obj).split('#')[-1].split('/')[-1]
            
            if not s.startswith('_') and not o.startswith('_'):
                triples.append((s, p, o))
        
        logger.info(f"Extracted {len(triples)} triples from graph")
        return triples
    
    def prepare_training_data(self, triples: List[Tuple[str, str, str]], 
                             test_ratio: float = 0.1):
        """Prepare triples for training and testing."""
        logger.info(f"Preparing {len(triples)} triples for training...")
        
        triples_array = np.array(triples)
        tf = TriplesFactory.from_labeled_triples(triples_array)
        training, testing = tf.split(ratios=[1.0 - test_ratio, test_ratio])
        
        logger.info(f"Training set: {len(training.triples)} triples")
        logger.info(f"Testing set: {len(testing.triples)} triples")
        logger.info(f"Entities: {training.num_entities}")
        logger.info(f"Relations: {training.num_relations}")
        
        return training, testing
    
    def train(self, graph, epochs: int = 100, embedding_dim: int = 128, 
              batch_size: int = 256):
        """Train TransE model on the graph."""
        logger.info("Starting TransE training...")
        
        # Stocker embedding_dim
        self.embedding_dim = embedding_dim
        
        triples = self.extract_triples_from_graph(graph)
        training_tf, testing_tf = self.prepare_training_data(triples)
        
        logger.info("Training TransE model...")
        result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model='TransE',
            model_kwargs=dict(
                embedding_dim=embedding_dim,
            ),
            optimizer='Adam',
            optimizer_kwargs=dict(
                lr=0.01,
            ),
            training_kwargs=dict(
                num_epochs=epochs,
                batch_size=batch_size,
            ),
            evaluation_kwargs=dict(
                batch_size=batch_size,
            ),
            random_seed=42,
        )
        
        self.model = result.model
        self.entity_to_id = training_tf.entity_to_id
        self.relation_to_id = training_tf.relation_to_id
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.trained = True
        
        logger.info("Training completed!")
        logger.info(f"Hits@10: {result.metric_results.get_metric('hits@10'):.4f}")
        logger.info(f"MRR: {result.metric_results.get_metric('mean_reciprocal_rank'):.4f}")
        
        self.save_model()
        
        return result
    
    def save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'model_state': self.model.state_dict(),
            'entity_to_id': self.entity_to_id,
            'relation_to_id': self.relation_to_id,
            'id_to_entity': self.id_to_entity,
            'embedding_dim': self.embedding_dim,  # Utiliser la valeur stockée
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        num_entities = len(model_data['entity_to_id'])
        num_relations = len(model_data['relation_to_id'])
        self.embedding_dim = model_data['embedding_dim']
        
        self.model = TransE(
            triples_factory=None,
            embedding_dim=self.embedding_dim,
        )
        
        self.model.entity_representations[0] = torch.nn.Embedding(
            num_entities, self.embedding_dim
        )
        self.model.relation_representations[0] = torch.nn.Embedding(
            num_relations, self.embedding_dim
        )
        
        self.model.load_state_dict(model_data['model_state'])
        self.entity_to_id = model_data['entity_to_id']
        self.relation_to_id = model_data['relation_to_id']
        self.id_to_entity = model_data['id_to_entity']
        self.trained = True
        
        logger.info("Model loaded successfully")
    
    def predict_for_user(self, user_id: str, relation: str = "hasRated", 
                        top_k: int = 12, exclude_movies: List[str] = None) -> List[Tuple[str, float]]:
        """Predict top-K movies for a user using link prediction."""
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        user_entity = user_id.split('#')[-1].split('/')[-1]
        
        if user_entity not in self.entity_to_id:
            logger.warning(f"User {user_entity} not in training data")
            return []
        
        if relation not in self.relation_to_id:
            logger.warning(f"Relation {relation} not found")
            return []
        
        user_idx = self.entity_to_id[user_entity]
        relation_idx = self.relation_to_id[relation]
        
        all_entity_ids = torch.arange(len(self.entity_to_id))
        
        batch_heads = torch.tensor([user_idx] * len(all_entity_ids))
        batch_relations = torch.tensor([relation_idx] * len(all_entity_ids))
        batch_tails = all_entity_ids
        
        with torch.no_grad():
            scores = self.model.score_hrt(
                torch.stack([batch_heads, batch_relations, batch_tails], dim=1)
            )
        
        scores = -scores.numpy()
        top_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        exclude_set = set(exclude_movies) if exclude_movies else set()
        
        for idx in top_indices:
            entity_id = self.id_to_entity[int(idx)]
            
            if entity_id.startswith('Movie') and entity_id not in exclude_set:
                score = float(scores[idx])
                recommendations.append((entity_id, score))
            
            if len(recommendations) >= top_k:
                break
        
        logger.info(f"Generated {len(recommendations)} link predictions for {user_entity}")
        return recommendations
    
    def get_embedding(self, entity_id: str) -> np.ndarray:
        """Get embedding vector for an entity."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        
        entity_clean = entity_id.split('#')[-1].split('/')[-1]
        
        if entity_clean not in self.entity_to_id:
            return None
        
        idx = self.entity_to_id[entity_clean]
        embedding = self.model.entity_representations[0].weight[idx].detach().numpy()
        
        return embedding