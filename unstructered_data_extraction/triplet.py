### Ce fichier a été généré par une IA generative

import csv
import re
from collections import defaultdict

def parse_triplets_file(input_file, output_file):
    print(input_file)

    """
    Parse le fichier de triplets et crée un CSV avec les données extraites.
    """
    movies_data = []
    current_movie = None
    current_data = {}
    all_predicates = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            print(line)
            
            # Détecte le début d'un nouveau film
            if line.startswith('FILM :'):
                # Sauvegarde le film précédent s'il existe
                if current_movie:
                    movies_data.append({'Title': current_movie, **current_data})
                    print(f"Saved movie: {current_movie} with data: {current_data}")
                
                # Initialise le nouveau film
                current_movie = line.replace('FILM :', '').strip()
                current_data = {}
            
            # Détecte les séparateurs
            elif line.startswith('---'):
                continue
            
            # Parse les triplets
            elif '|' in line and current_movie:
                parts = line.split('|')
                if len(parts) >= 3:
                    # Le titre du film
                    film_name = parts[0].strip()
                    # Le prédicat
                    predicate = parts[1].strip()
                    # La valeur
                    value = '|'.join(parts[2:]).strip()
                    
                    # Nettoie les valeurs "not specified" ou similaires
                    if is_not_specified(value):
                        value = ''
                    
                    # Ajoute le prédicat à l'ensemble
                    all_predicates.add(predicate)
                    
                    # Stocke la valeur
                    current_data[predicate] = value
        
        # N'oublie pas le dernier film
        if current_movie:
            movies_data.append({'Title': current_movie, **current_data})
    
    if movies_data:
        fieldnames = ['Title'] + sorted(all_predicates)
        
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for movie in movies_data:
                row = {field: movie.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"✓ Extraction terminée !")
        print(f"✓ {len(movies_data)} films extraits")
        print(f"✓ {len(all_predicates)} prédicats trouvés")
        print(f"✓ Fichier créé : {output_file}")
    else:
        print(input_file)
        print("✗ Aucune donnée trouvée dans le fichier")

def is_not_specified(value):
    """
    Vérifie si la valeur indique qu'elle n'est pas spécifiée.
    """
    if not value:
        return True
    
    value_lower = value.lower()
    not_specified_patterns = [
        'not specified',
        'not mentioned',
        'none',
        '(not specified',
        'not specified in',
        'as of provided context'
    ]
    
    for pattern in not_specified_patterns:
        if pattern in value_lower:
            return True
    
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'triplets_extraction_movies.txt'
        print(f"⚠️  Aucun fichier d'entrée spécifié. Utilisation du fichier par défaut : {input_file}\n")
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'movies_data.csv'
    
    print("Début de l'extraction...\n")
    parse_triplets_file(input_file, output_file)