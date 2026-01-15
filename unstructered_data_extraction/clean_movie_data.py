import pandas as pd


df = pd.read_csv('movies_with_ids.csv')

# 2. Nettoyage et conversion de la colonne 'id'
# On supprime les espaces éventuels et on convertit en numérique
df['id'] = pd.to_numeric(df['id'], errors='coerce')

# 3. Conversion en type Integer (Int64 permet de gérer les NaN)
df['id'] = df['id'].astype('Int64')

# 4. Tri par ID (optionnel, mais utile pour l'organisation)
df = df.sort_values(by='id', ascending=True)

# 5. Sauvegarde du fichier propre
df.to_csv('movies_cleaned_final.csv', index=False)

print("Traitement terminé. Voici un aperçu des données :")
print(df[['id', 'Title']].head())