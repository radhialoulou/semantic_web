import csv

print("Debut du traitement...")

with open('first_iteration_credit.csv', 'r', encoding='utf-8') as f:
    lecteur = csv.DictReader(f)
    cast_lignes = []
    crew_lignes = []
    
    nb_lignes = 0
    for ligne in lecteur:
        nb_lignes += 1
        
        if ligne['type'] == 'cast':
            cast_lignes.append({
                'id': ligne['id'],
                'name': ligne['name'],
                'character': ligne['character'],
                'order': ligne['order'],
                'gender': ligne['gender']
            })
        elif ligne['type'] == 'crew':
            crew_lignes.append({
                'id': ligne['id'],
                'name': ligne['name'],
                'job': ligne['job'],
                'department': ligne['department'],
                'gender': ligne['gender']
            })
        
        if nb_lignes % 1000 == 0:
            print(f"Lignes traitees: {nb_lignes}")

print(f"Total lignes traitees: {nb_lignes}")
print(f"Cast entries: {len(cast_lignes)}")
print(f"Crew entries: {len(crew_lignes)}")

cast_cols = ['id', 'name', 'character', 'order', 'gender']
crew_cols = ['id', 'name', 'job', 'department', 'gender']

with open('cast.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=cast_cols)
    writer.writeheader()
    writer.writerows(cast_lignes)

print("Fichier cast.csv cree!")

with open('crew.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=crew_cols)
    writer.writeheader()
    writer.writerows(crew_lignes)

print("Fichier crew.csv cree!")
print("Termine!")