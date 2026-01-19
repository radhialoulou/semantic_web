
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)



def jaccard_similarity(set_a, set_b):
    """
    Calcule la similarité de Jaccard entre deux ensembles.
    
    Formule: J(A,B) = |A ∩ B| / |A ∪ B|
    
    Args:
        set_a: Premier ensemble (ex: films notés par utilisateur A)
        set_b: Deuxième ensemble (ex: films notés par utilisateur B)
    
    Returns:
        float: Score entre 0 et 1 (0 = aucune similarité, 1 = identiques)
    
    Référence:
        Jaccard, P. (1912). "The distribution of the flora in the alpine zone"
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0
    
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)


def compute_jaccard_user_similarity(user1_movies, user2_movies):
    """
    Calcule la similarité Jaccard entre deux utilisateurs basée sur leurs films notés.
    
    Args:
        user1_movies: set de movie IDs notés par user1
        user2_movies: set de movie IDs notés par user2
    
    Returns:
        float: Score de similarité Jaccard
    """
    return jaccard_similarity(user1_movies, user2_movies)


# =============================================================================
# 2. COSINE SIMILARITY
# =============================================================================

def cosine_similarity(vector_a, vector_b):
    """
    Calcule la similarité cosinus entre deux vecteurs.
    
    Formule: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vector_a: Premier vecteur (ex: vecteur de notes de l'utilisateur A)
        vector_b: Deuxième vecteur (ex: vecteur de notes de l'utilisateur B)
    
    Returns:
        float: Score entre -1 et 1 (1 = même direction, -1 = opposés)
    
    Référence:
        Salton, G., & McGill, M. J. (1983). "Introduction to modern information retrieval"
    """
    if len(vector_a) == 0 or len(vector_b) == 0:
        return 0.0
    
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def compute_cosine_user_similarity(user1_ratings, user2_ratings, common_movies):
    """
    Calcule la similarité cosinus entre deux utilisateurs sur leurs films communs.
    
    Args:
        user1_ratings: dict {movie_id: rating} pour user1
        user2_ratings: dict {movie_id: rating} pour user2
        common_movies: list des movie IDs en commun
    
    Returns:
        float: Score de similarité cosinus
    """
    if len(common_movies) == 0:
        return 0.0
    
    vector1 = np.array([user1_ratings[movie] for movie in common_movies])
    vector2 = np.array([user2_ratings[movie] for movie in common_movies])
    
    return cosine_similarity(vector1, vector2)


# =============================================================================
# 3. PEARSON CORRELATION
# =============================================================================

def pearson_correlation(ratings_a, ratings_b):
    """
    Calcule la corrélation de Pearson entre deux séries de notes.
    
    Formule: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
    
    Args:
        ratings_a: Liste de notes de l'utilisateur A
        ratings_b: Liste de notes de l'utilisateur B (même taille que ratings_a)
    
    Returns:
        float: Coefficient entre -1 et 1 (1 = corrélation positive parfaite)
    
    Référence:
        Pearson, K. (1895). "Notes on regression and inheritance in the case of two parents"
    """
    if len(ratings_a) < 2 or len(ratings_b) < 2:
        return 0.0
    
    mean_a = np.mean(ratings_a)
    mean_b = np.mean(ratings_b)
    
    numerator = np.sum((ratings_a - mean_a) * (ratings_b - mean_b))
    denominator = np.sqrt(np.sum((ratings_a - mean_a)**2) * np.sum((ratings_b - mean_b)**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_pearson_user_similarity(user1_ratings, user2_ratings, common_movies):
    """
    Calcule la corrélation de Pearson entre deux utilisateurs sur leurs films communs.
    
    Args:
        user1_ratings: dict {movie_id: rating} pour user1
        user2_ratings: dict {movie_id: rating} pour user2
        common_movies: list des movie IDs en commun
    
    Returns:
        float: Coefficient de corrélation de Pearson
    """
    if len(common_movies) < 2:
        return 0.0
    
    ratings1 = np.array([user1_ratings[movie] for movie in common_movies])
    ratings2 = np.array([user2_ratings[movie] for movie in common_movies])
    
    return pearson_correlation(ratings1, ratings2)


# =============================================================================
# 4. K-NEAREST NEIGHBORS (KNN)
# =============================================================================

def find_k_nearest_neighbors(target_user_id, all_users_data, k=10, similarity_method='jaccard'):
    """
    Trouve les K utilisateurs les plus similaires à l'utilisateur cible.
    
    Args:
        target_user_id: ID de l'utilisateur cible
        all_users_data: dict {user_id: {'movies': set(), 'ratings': dict()}}
        k: Nombre de voisins à retourner
        similarity_method: 'jaccard', 'cosine', ou 'pearson'
    
    Returns:
        list: [(user_id, similarity_score), ...] triée par similarité décroissante
    
    Référence:
        Resnick, P., et al. (1994). "GroupLens: an open architecture for collaborative filtering"
    """
    target_data = all_users_data.get(target_user_id)
    if not target_data:
        return []
    
    target_movies = target_data['movies']
    target_ratings = target_data['ratings']
    
    similarities = []
    
    for user_id, user_data in all_users_data.items():
        if user_id == target_user_id:
            continue
        
        user_movies = user_data['movies']
        user_ratings = user_data['ratings']
        
        # Calculer la similarité selon la méthode choisie
        if similarity_method == 'jaccard':
            similarity = compute_jaccard_user_similarity(target_movies, user_movies)
        
        elif similarity_method == 'cosine':
            common_movies = list(target_movies.intersection(user_movies))
            similarity = compute_cosine_user_similarity(target_ratings, user_ratings, common_movies)
        
        elif similarity_method == 'pearson':
            common_movies = list(target_movies.intersection(user_movies))
            similarity = compute_pearson_user_similarity(target_ratings, user_ratings, common_movies)
        
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
        
        if similarity > 0:
            similarities.append((user_id, similarity))
    
    # Trier par similarité décroissante et prendre les K premiers
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]


# =============================================================================
# 5. COLLABORATIVE FILTERING PREDICTION
# =============================================================================

def predict_rating_user_based(target_user_id, movie_id, all_users_data, k=10, similarity_method='pearson'):
    """
    Prédit la note qu'un utilisateur donnerait à un film en utilisant le filtrage collaboratif.
    
    Formule: r̂(u,i) = r̄u + Σ[sim(u,v) × (r(v,i) - r̄v)] / Σ|sim(u,v)|
    
    Args:
        target_user_id: ID de l'utilisateur
        movie_id: ID du film
        all_users_data: Données de tous les utilisateurs
        k: Nombre de voisins à utiliser
        similarity_method: Méthode de similarité ('jaccard', 'cosine', 'pearson')
    
    Returns:
        float: Note prédite (ou None si impossible de prédire)
    
    Référence:
        Breese, J. S., et al. (1998). "Empirical analysis of predictive algorithms for collaborative filtering"
    """
    # Trouver les K voisins les plus proches
    neighbors = find_k_nearest_neighbors(target_user_id, all_users_data, k, similarity_method)
    
    if len(neighbors) == 0:
        return None
    
    target_data = all_users_data[target_user_id]
    target_avg = np.mean(list(target_data['ratings'].values())) if target_data['ratings'] else 0
    
    numerator = 0
    denominator = 0
    
    for neighbor_id, similarity in neighbors:
        neighbor_data = all_users_data[neighbor_id]
        
        # Vérifier si le voisin a noté ce film
        if movie_id in neighbor_data['ratings']:
            neighbor_rating = neighbor_data['ratings'][movie_id]
            neighbor_avg = np.mean(list(neighbor_data['ratings'].values()))
            
            # Formule de prédiction pondérée
            numerator += similarity * (neighbor_rating - neighbor_avg)
            denominator += abs(similarity)
    
    if denominator == 0:
        return None
    
    predicted_rating = target_avg + (numerator / denominator)
    
    # Borner la prédiction entre 0.5 et 5.0 (échelle MovieLens)
    return max(0.5, min(5.0, predicted_rating))


# =============================================================================
# 6. HYBRID SCORING
# =============================================================================

def compute_hybrid_score(content_score, collaborative_score, user_ratings_count, alpha=0.5):
    """
    Combine les scores content-based et collaborative filtering de manière adaptative.
    
    Args:
        content_score: Score basé sur le contenu (0-1)
        collaborative_score: Score basé sur le filtrage collaboratif (0-1)
        user_ratings_count: Nombre de films notés par l'utilisateur
        alpha: Paramètre de pondération (0 = full collaborative, 1 = full content)
    
    Returns:
        float: Score hybride (0-100)
    
    Référence:
        Burke, R. (2002). "Hybrid recommender systems: Survey and experiments"
    """
    # Ajustement adaptatif du alpha selon le nombre de notes
    if user_ratings_count < 5:  # Cold start
        adaptive_alpha = 0.7  # Plus de poids au content-based
    elif user_ratings_count < 15:  # Active user
        adaptive_alpha = alpha  # Équilibre
    else:  # Expert user
        adaptive_alpha = 0.6  # Légèrement plus de content pour affiner
    
    hybrid_score = adaptive_alpha * content_score + (1 - adaptive_alpha) * collaborative_score
    
    return hybrid_score * 100  # Convertir en pourcentage


# =============================================================================
# 7. UTILITY FUNCTIONS
# =============================================================================

def load_users_data_structure(ratings_data):
    """
    Transforme les données de ratings en structure optimisée pour les calculs de similarité.
    
    Args:
        ratings_data: Liste de tuples (user_id, movie_id, rating)
    
    Returns:
        dict: {user_id: {'movies': set(movie_ids), 'ratings': {movie_id: rating}}}
    """
    users_data = defaultdict(lambda: {'movies': set(), 'ratings': {}})
    
    for user_id, movie_id, rating in ratings_data:
        users_data[user_id]['movies'].add(movie_id)
        users_data[user_id]['ratings'][movie_id] = rating
    
    return dict(users_data)


def evaluate_similarity_methods(users_data, sample_size=100):
    """
    Compare les différentes méthodes de similarité sur un échantillon d'utilisateurs.
    Utile pour analyser quelle méthode fonctionne le mieux sur vos données.
    
    Args:
        users_data: Structure de données utilisateurs
        sample_size: Nombre de paires d'utilisateurs à comparer
    
    Returns:
        dict: Statistiques comparatives des méthodes
    """
    import random
    
    user_ids = list(users_data.keys())
    stats = {
        'jaccard': [],
        'cosine': [],
        'pearson': []
    }
    
    for _ in range(min(sample_size, len(user_ids) * (len(user_ids) - 1) // 2)):
        u1, u2 = random.sample(user_ids, 2)
        
        u1_movies = users_data[u1]['movies']
        u2_movies = users_data[u2]['movies']
        common = list(u1_movies.intersection(u2_movies))
        
        if len(common) >= 2:  # Besoin d'au moins 2 films en commun
            # Jaccard
            jaccard_sim = compute_jaccard_user_similarity(u1_movies, u2_movies)
            stats['jaccard'].append(jaccard_sim)
            
            # Cosine
            cosine_sim = compute_cosine_user_similarity(
                users_data[u1]['ratings'],
                users_data[u2]['ratings'],
                common
            )
            stats['cosine'].append(cosine_sim)
            
            # Pearson
            pearson_sim = compute_pearson_user_similarity(
                users_data[u1]['ratings'],
                users_data[u2]['ratings'],
                common
            )
            stats['pearson'].append(pearson_sim)
    
    return {
        method: {
            'mean': np.mean(scores) if scores else 0,
            'std': np.std(scores) if scores else 0,
            'min': np.min(scores) if scores else 0,
            'max': np.max(scores) if scores else 0
        }
        for method, scores in stats.items()
    }


if __name__ == "__main__":
    # Test simple
    print("Testing Jaccard Similarity...")
    set_a = {1, 2, 3, 4, 5}
    set_b = {3, 4, 5, 6, 7}
    print(f"Jaccard({set_a}, {set_b}) = {jaccard_similarity(set_a, set_b):.3f}")
    
    print("\nTesting Cosine Similarity...")
    vec_a = np.array([4, 5, 3, 4, 5])
    vec_b = np.array([3, 4, 5, 4, 3])
    print(f"Cosine = {cosine_similarity(vec_a, vec_b):.3f}")
    
    print("\nTesting Pearson Correlation...")
    print(f"Pearson = {pearson_correlation(vec_a, vec_b):.3f}")