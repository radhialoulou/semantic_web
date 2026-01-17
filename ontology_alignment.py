"""
Semi-Automatic Ontology Alignment Tool
=======================================
This script analyzes the movie ontology and proposes alignments with:
- Schema.org (https://schema.org/)
- DBpedia Ontology (http://dbpedia.org/ontology/)
- Wikidata (https://www.wikidata.org/)

It uses string similarity and semantic matching to suggest owl:equivalentClass
and owl:equivalentProperty mappings.

Author: Sara Aymeric Radhi
Date: 2026-01-16
"""

import rdflib
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDFS, RDF
from difflib import SequenceMatcher
from typing import List, Dict, Tuple
import json

# Namespaces
MOVIE_NS = Namespace("http://saraaymericradhi.org/movie-ontology#")
SCHEMA = Namespace("http://schema.org/")
DBO = Namespace("http://dbpedia.org/ontology/")
DBP = Namespace("http://dbpedia.org/property/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# ============================================================================
# KNOWLEDGE BASE: Schema.org and DBpedia vocabularies
# ============================================================================

SCHEMA_CLASSES = {
    "Movie": "A movie or film",
    "Person": "A person",
    "Organization": "An organization",
    "MusicRecording": "A music recording/song",
    "Rating": "A rating/review",
    "Award": "An award or honor",
    "CreativeWork": "A creative work",
    "Event": "An event",
    "Place": "A place/location",
}

SCHEMA_PROPERTIES = {
    "name": "The name of the item",
    "director": "The director of the movie",
    "actor": "An actor in the movie",
    "author": "The author/writer",
    "genre": "Genre of the work",
    "budget": "The budget",
    "description": "A description",
    "datePublished": "Date of publication/release",
    "productionCompany": "The production company",
    "musicBy": "Music composer",
    "duration": "Duration/runtime",
    "boxOffice": "Box office revenue",
    "ratingValue": "The rating value",
}

DBPEDIA_CLASSES = {
    "Film": "A movie or film",
    "Person": "A person",
    "Organisation": "An organization",
    "MusicalWork": "A musical work",
    "MusicalArtist": "A musical artist/composer",
    "Award": "An award",
    "TopicalConcept": "A topical concept",
}

DBPEDIA_PROPERTIES = {
    "director": "The director",
    "starring": "Main actors/stars",
    "writer": "The writer",
    "genre": "Genre",
    "budget": "Budget",
    "abstract": "Abstract/description",
    "releaseDate": "Release date",
    "productionCompany": "Production company",
    "musicComposer": "Music composer",
    "runtime": "Runtime/duration",
    "gross": "Box office gross",
}

# ============================================================================
# STRING SIMILARITY
# ============================================================================

def similarity(a: str, b: str) -> float:
    """Calculate string similarity ratio (0-1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def semantic_match(term: str, keywords: List[str]) -> float:
    """Check if term semantically matches any keyword"""
    term_lower = term.lower()
    for keyword in keywords:
        if keyword.lower() in term_lower or term_lower in keyword.lower():
            return 1.0
    return 0.0

# ============================================================================
# ALIGNMENT SUGGESTION ENGINE
# ============================================================================

class AlignmentSuggester:
    """Semi-automatic ontology alignment engine"""
    
    def __init__(self, ontology_path: str):
        self.graph = Graph()
        self.graph.parse(ontology_path, format="turtle")
        self.suggestions = []
        
    def extract_local_name(self, uri: URIRef) -> str:
        """Extract local name from URI"""
        return str(uri).split('#')[-1].split('/')[-1]
    
    def suggest_class_alignments(self, threshold: float = 0.6) -> List[Dict]:
        """Suggest class alignments based on string similarity"""
        suggestions = []
        
        # Get all classes from our ontology
        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if str(cls).startswith(str(MOVIE_NS)):
                local_name = self.extract_local_name(cls)
                
                # Check Schema.org
                for schema_cls, desc in SCHEMA_CLASSES.items():
                    score = similarity(local_name, schema_cls)
                    if score >= threshold:
                        suggestions.append({
                            "type": "class",
                            "source": str(cls),
                            "source_label": local_name,
                            "target": str(SCHEMA[schema_cls]),
                            "target_vocab": "schema.org",
                            "target_label": schema_cls,
                            "confidence": score,
                            "relation": "owl:equivalentClass"
                        })
                
                # Check DBpedia
                for dbo_cls, desc in DBPEDIA_CLASSES.items():
                    score = similarity(local_name, dbo_cls)
                    if score >= threshold:
                        suggestions.append({
                            "type": "class",
                            "source": str(cls),
                            "source_label": local_name,
                            "target": str(DBO[dbo_cls]),
                            "target_vocab": "DBpedia",
                            "target_label": dbo_cls,
                            "confidence": score,
                            "relation": "owl:equivalentClass"
                        })
        
        return suggestions
    
    def suggest_property_alignments(self, threshold: float = 0.6) -> List[Dict]:
        """Suggest property alignments"""
        suggestions = []
        
        # Get all properties from our ontology
        for prop in set(self.graph.subjects(RDF.type, OWL.ObjectProperty)) | \
                     set(self.graph.subjects(RDF.type, OWL.DatatypeProperty)):
            if str(prop).startswith(str(MOVIE_NS)):
                local_name = self.extract_local_name(prop)
                
                # Remove "has" prefix for better matching
                clean_name = local_name.replace("has", "").replace("Has", "")
                
                # Check Schema.org
                for schema_prop, desc in SCHEMA_PROPERTIES.items():
                    score = max(
                        similarity(local_name, schema_prop),
                        similarity(clean_name, schema_prop)
                    )
                    if score >= threshold:
                        suggestions.append({
                            "type": "property",
                            "source": str(prop),
                            "source_label": local_name,
                            "target": str(SCHEMA[schema_prop]),
                            "target_vocab": "schema.org",
                            "target_label": schema_prop,
                            "confidence": score,
                            "relation": "owl:equivalentProperty"
                        })
                
                # Check DBpedia
                for dbo_prop, desc in DBPEDIA_PROPERTIES.items():
                    score = max(
                        similarity(local_name, dbo_prop),
                        similarity(clean_name, dbo_prop)
                    )
                    if score >= threshold:
                        suggestions.append({
                            "type": "property",
                            "source": str(prop),
                            "source_label": local_name,
                            "target": str(DBO[dbo_prop]),
                            "target_vocab": "DBpedia",
                            "target_label": dbo_prop,
                            "confidence": score,
                            "relation": "owl:equivalentProperty"
                        })
        
        return suggestions
    
    def generate_alignment_report(self, output_file: str = "alignment_report.json"):
        """Generate a comprehensive alignment report"""
        print("Analyzing ontology for alignment opportunities...")
        print("=" * 70)
        
        class_suggestions = self.suggest_class_alignments(threshold=0.5)
        property_suggestions = self.suggest_property_alignments(threshold=0.5)
        
        all_suggestions = class_suggestions + property_suggestions
        
        # Sort by confidence
        all_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nALIGNMENT SUGGESTIONS SUMMARY")
        print(f"   Classes: {len(class_suggestions)} suggestions")
        print(f"   Properties: {len(property_suggestions)} suggestions")
        print(f"   Total: {len(all_suggestions)} suggestions\n")
        
        # Display top suggestions
        print("TOP ALIGNMENT SUGGESTIONS:\n")
        for i, sugg in enumerate(all_suggestions[:20], 1):
            print(f"{i}. {sugg['type'].upper()}: {sugg['source_label']}")
            print(f"   -> {sugg['target_vocab']}: {sugg['target_label']}")
            print(f"   Confidence: {sugg['confidence']:.2%}")
            print(f"   Relation: {sugg['relation']}")
            print()
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_suggestions": len(all_suggestions),
                    "class_alignments": len(class_suggestions),
                    "property_alignments": len(property_suggestions)
                },
                "suggestions": all_suggestions
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Full report saved to: {output_file}")
        
        return all_suggestions
    
    def generate_turtle_alignments(self, suggestions: List[Dict], 
                                   min_confidence: float = 0.7,
                                   output_file: str = "suggested_alignments.ttl"):
        """Generate Turtle file with suggested alignments"""
        
        # Filter by confidence
        filtered = [s for s in suggestions if s['confidence'] >= min_confidence]
        
        turtle_lines = [
            "@prefix : <http://saraaymericradhi.org/movie-ontology#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix schema: <http://schema.org/> .",
            "@prefix dbo: <http://dbpedia.org/ontology/> .",
            "@prefix dbp: <http://dbpedia.org/property/> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "",
            "# Semi-Automatic Ontology Alignment Suggestions",
            f"# Generated: 2026-01-16",
            f"# Minimum confidence: {min_confidence:.0%}",
            f"# Total suggestions: {len(filtered)}",
            "",
        ]
        
        # Group by type
        class_alignments = [s for s in filtered if s['type'] == 'class']
        property_alignments = [s for s in filtered if s['type'] == 'property']
        
        if class_alignments:
            turtle_lines.append("# ========== CLASS ALIGNMENTS ==========")
            turtle_lines.append("")
            for sugg in class_alignments:
                source_local = self.extract_local_name(URIRef(sugg['source']))
                turtle_lines.append(f":{source_local} owl:equivalentClass <{sugg['target']}> .")
                turtle_lines.append(f"    # Confidence: {sugg['confidence']:.2%} - {sugg['target_vocab']}")
                turtle_lines.append("")
        
        if property_alignments:
            turtle_lines.append("# ========== PROPERTY ALIGNMENTS ==========")
            turtle_lines.append("")
            for sugg in property_alignments:
                source_local = self.extract_local_name(URIRef(sugg['source']))
                turtle_lines.append(f":{source_local} owl:equivalentProperty <{sugg['target']}> .")
                turtle_lines.append(f"    # Confidence: {sugg['confidence']:.2%} - {sugg['target_vocab']}")
                turtle_lines.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(turtle_lines))
        
        print(f"\nTurtle alignments (>={min_confidence:.0%} confidence) saved to: {output_file}")
        print(f"   {len(filtered)} alignments written")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  SEMI-AUTOMATIC ONTOLOGY ALIGNMENT TOOL")
    print("  Movie Ontology -> Schema.org, DBpedia, Wikidata")
    print("=" * 70)
    print()
    
    # Initialize suggester
    aligner = AlignmentSuggester("ontology/ontology.ttl")
    
    # Generate comprehensive report
    suggestions = aligner.generate_alignment_report("ontology/alignment_report.json")
    
    # Generate high-confidence Turtle alignments
    aligner.generate_turtle_alignments(
        suggestions, 
        min_confidence=0.7,
        output_file="ontology/suggested_alignments.ttl"
    )
    
    print("\n" + "=" * 70)
    print("ALIGNMENT ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review alignment_report.json for all suggestions")
    print("2. Review suggested_alignments.ttl for high-confidence alignments")
    print("3. Manually validate and integrate approved alignments into ontology.ttl")
