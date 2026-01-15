import pandas as pd

df = pd.read_csv('movies_data.csv')

columns_to_keep = [
    'Title',                                    
    'hasFrenchTitle', 'hasFrenchQuebecTitle',    
    'hasFrenchReleaseDate', 'hasFrenchVoiceCast', 
    'hasFrenchAdaptor', 'hasFrenchAdaptationYear',
    'hasAwardName', 'hasAwardYear',             
    'hasBoxOfficeRevenue', 
    'hasDirector', 'hasProducer', 'hasScenarioWriters', 
    'hasComposer',                               
    'hasThemeSongTitle', 'hasThemeSongCompositor'
]


df_filtered = df[df.columns.intersection(columns_to_keep)]

# Affichage du résultat
print(df_filtered.head())

# Sauvegarde du nouveau fichier filtré
df_filtered.to_csv('movie_data_llm.csv', index=False)