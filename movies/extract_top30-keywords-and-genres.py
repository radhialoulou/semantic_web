import csv
import ast
from collections import Counter

# Configuration
MOVIES_METADATA_FILE = 'movies_metadata_top200.csv'
KEYWORDS_FILE = 'keywords_top200.csv'
OUTPUT_DIR = '../ontology/'
TOP_N = 30


def extract_genres(filepath):
    """Extract all genres from movies metadata CSV file."""
    genres_counter = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                genres_list = ast.literal_eval(row['genres'])
                for genre in genres_list:
                    genres_counter[(genre['id'], genre['name'])] += 1
            except (ValueError, SyntaxError, KeyError):
                continue
    
    return genres_counter


def extract_keywords(filepath):
    """Extract all keywords from keywords CSV file."""
    keywords_counter = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keywords_list = ast.literal_eval(row['keywords'])
                for keyword in keywords_list:
                    keywords_counter[(keyword['id'], keyword['name'])] += 1
            except (ValueError, SyntaxError, KeyError):
                continue
    
    return keywords_counter


def save_to_csv(data, filename, headers):
    """Save extracted data to CSV file."""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for (item_id, item_name), count in data:
            writer.writerow([item_id, item_name, count])
    print(f"Saved to: {filename}")


def main():
    """Main function to extract and display top genres and keywords."""
    # Extract genres
    print(f"TOP {TOP_N} GENRES")
    
    genres_counter = extract_genres(MOVIES_METADATA_FILE)
    print(f"{'ID':<10} {'Genre':<25} {'Count':<10}")
    print("-" * 45)
    top_genres = genres_counter.most_common(TOP_N)
    for (gid, gname), count in top_genres:
        print(f"{gid:<10} {gname:<25} {count:<10}")
    
    # Extract keywords
    print()
    print(f"TOP {TOP_N} KEYWORDS")
    
    keywords_counter = extract_keywords(KEYWORDS_FILE)
    print(f"{'ID':<10} {'Keyword':<35} {'Count':<10}")
    print("-" * 55)
    top_keywords = keywords_counter.most_common(TOP_N)
    for (kid, kname), count in top_keywords:
        print(f"{kid:<10} {kname:<35} {count:<10}")
    
    print()
    print(f"Total unique genres: {len(genres_counter)}")
    print(f"Total unique keywords: {len(keywords_counter)}")
    
    # Save results to CSV files
    print()
    print("SAVING RESULTS TO FILES")
    
    save_to_csv(top_genres, OUTPUT_DIR + 'extracted_genres.csv', ['id', 'name', 'count'])
    save_to_csv(top_keywords, OUTPUT_DIR + 'extracted_keywords.csv', ['id', 'name', 'count'])
    
    print()
    print("Extraction complete!")


if __name__ == "__main__":
    main()
