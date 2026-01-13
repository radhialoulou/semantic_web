# Script PowerShell pour exécuter la transformation RML complète
# Usage: .\run_rml.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TRANSFORMATION RML - Movies Ontology" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Étape 1: Normaliser les données CSV
Write-Host "`n[1/2] Normalisation des données CSV..." -ForegroundColor Yellow
python normalize_movies.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Erreur lors de la normalisation!" -ForegroundColor Red
    exit 1
}

# Étape 2: Exécuter RMLMapper
Write-Host "`n[2/2] Exécution de RMLMapper..." -ForegroundColor Yellow

# Vérifier si rmlmapper.jar existe
if (Test-Path "rmlmapper.jar") {
    java -jar rmlmapper.jar -m movies_mapping.rml.ttl -o movies_output.ttl -s turtle
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n============================================" -ForegroundColor Green
        Write-Host "  TRANSFORMATION TERMINÉE AVEC SUCCÈS!" -ForegroundColor Green
        Write-Host "============================================" -ForegroundColor Green
        Write-Host "Fichier de sortie: movies_output.ttl"
    } else {
        Write-Host "Erreur lors de l'exécution de RMLMapper!" -ForegroundColor Red
    }
} else {
    Write-Host "`nATTENTION: rmlmapper.jar non trouvé!" -ForegroundColor Red
    Write-Host "Téléchargez-le depuis: https://github.com/RMLio/rmlmapper-java/releases" -ForegroundColor Yellow
    Write-Host "`nPuis exécutez manuellement:" -ForegroundColor Yellow
    Write-Host "  java -jar rmlmapper.jar -m movies_mapping.rml.ttl -o movies_output.ttl -s turtle"
}
