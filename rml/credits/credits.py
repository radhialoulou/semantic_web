import csv

with open('./../movies/credits_top200.csv', 'r', encoding='utf-8') as f:
    lecteur = csv.DictReader(f)
    lignes = []
    
    for ligne in lecteur:
        if all(v == '' for v in ligne.values()):
            continue
        
        film_id = ligne.get('id', '').strip().strip('"')
        
        for col, val in ligne.items():
            if col == 'id' or val == '' or val == '""':
                continue
            
            val_clean = val.strip().strip('"')
            
            if val_clean.startswith('[{'):
                try:
                    data = eval(val_clean)
                    for item in data:
                        nouvelle_ligne = {'id': film_id}
                        
                        if col == 'cast':
                            nouvelle_ligne['type'] = 'cast'
                            nouvelle_ligne['name'] = item.get('name', '')
                            nouvelle_ligne['character'] = item.get('character', '')
                            nouvelle_ligne['order'] = item.get('order', '')
                            nouvelle_ligne['gender'] = item.get('gender', '')
                        elif col == 'crew':
                            nouvelle_ligne['type'] = 'crew'
                            nouvelle_ligne['name'] = item.get('name', '')
                            nouvelle_ligne['job'] = item.get('job', '')
                            nouvelle_ligne['department'] = item.get('department', '')
                            nouvelle_ligne['gender'] = item.get('gender', '')
                        
                        lignes.append(nouvelle_ligne)
                except:
                    pass

colonnes = ['id', 'type', 'name', 'character', 'order', 'job', 'department', 'gender']

with open('output.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=colonnes)
    writer.writeheader()
    writer.writerows(lignes)