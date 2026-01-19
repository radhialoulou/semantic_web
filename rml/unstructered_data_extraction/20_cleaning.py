import urllib.parse

def process_ttl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. On décode d'abord les caractères spéciaux (%2F -> /, etc.)
    # Mais on garde les %20 temporairement pour ne pas les mélanger avec de vrais espaces
    decoded_content = urllib.parse.unquote(content)
    
    # 2. On remplace les espaces (qui étaient des %20) par des underscores
    # On cible uniquement ce qui est entre < > pour ne pas casser les labels "normaux"
    import re
    def replace_spaces_in_uris(match):
        return match.group(0).replace(' ', '_')
    
    # Cette regex trouve tout ce qui est entre chevrons <http://...>
    cleaned_content = re.sub(r'<(.*?)>', replace_spaces_in_uris, decoded_content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"Nettoyage terminé. Les espaces dans les URIs ont été remplacés par '_' dans : {output_file}")

# Utilisation
process_ttl_file('./../credits/output.ttl', './../credits/output_cleaned.ttl')