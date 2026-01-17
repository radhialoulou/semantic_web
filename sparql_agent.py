import re
from typing import List, Dict, Any
import rdflib
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from graph_loader import load_graph

# Configure Local LLM (Ollama)
# User specified model: gpt-oss:20b
LLM = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Required but ignored
    model="gpt-oss:20b",
    temperature=0
)

# Complete Ontology Schema extracted from ontology.ttl
# This comprehensive schema helps the LLM generate accurate SPARQL queries
ONTOLOGY_SCHEMA = """
## PREFIXES
@prefix : <http://saraaymericradhi.org/movie-ontology#>
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>
@prefix owl: <http://www.w3.org/2002/07/owl#>
@prefix genres: <http://saraaymericradhi.org/movie-ontology/genres#>
@prefix foaf: <http://xmlns.com/foaf/0.1/>
@prefix schema: <http://schema.org/>

## MAIN CLASSES

### Movie Classes
:Movie - A motion picture/film
  Subclasses:
    :AdultMovie - Movies for adults (18+)
    :TeenMovie - Movies for teenagers (13-17)
    :ChildrenMovie - Movies for children (<13)
    :Blockbuster - Movies with revenue >= $100M
    :HighlyRatedMovie - Movies with rating >= 8.0
    :Masterpiece - Blockbuster AND HighlyRatedMovie
    :ProfitableMovie - Revenue > Budget
    :AwardWinningMovie - Has won at least one award
    :OscarWinner - Has won at least one Oscar

### Movie Component Classes (All Disjoint)
:MovieInformation - Basic info (budget, date, language, runtime)
:MovieResult - Performance metrics (ratings, revenue, votes)
:MovieProduction - Production info (cast, crew, companies, countries)
:MovieContent - Content description (genres, keywords, plot, collection)

### Person Classes
:Person - A person involved in movie production
  Subclasses:
    :CastMember - Actor/actress in a movie (DISJOINT with CrewMember)
      :LeadActor - CastMember with castOrder = 0
    :CrewMember - Technical/production crew member
      :Director - Person who directs movies
      :Writer - Screenplay writer
    :VoiceActor - Voice actor for dubbed versions
    :Composer - Music composer
    :User - User who rates movies

### Other Classes
:Character - Fictional character in a movie (DISJOINT with Person)
:Genre - Movie genre (Action, Drama, etc.)
:Keyword - Descriptive keyword for content
:Collection - Movie collection/franchise
:Rating - User rating for a movie
:Award - Prize/award given to movies
:Song - Musical piece/soundtrack
:FrenchAdaptation - French version/dubbing info

## PROPERTIES

### Movie Direct Properties
:hasTitle (domain: Movie, range: xsd:string) - FUNCTIONAL, INVERSE FUNCTIONAL
:hasTagline (domain: Movie, range: xsd:string)
:hasInformation (domain: Movie, range: MovieInformation)
:hasResult (domain: Movie, range: MovieResult)
:hasProduction (domain: Movie, range: MovieProduction)
:hasContent (domain: Movie, range: MovieContent)
:hasWikipediaPage (domain: owl:Thing, range: foaf:Document)
:hasRating (domain: Movie, range: Rating)
:hasAward (domain: Movie, range: Award)
:hasFrenchAdaptation (domain: Movie, range: FrenchAdaptation)
:hasThemeSong (domain: Movie, range: Song)

### MovieInformation Properties
:hasBudget (domain: MovieInformation, range: xsd:float) - FUNCTIONAL
:hasReleaseDate (domain: MovieInformation, range: xsd:date) - FUNCTIONAL
:hasLanguage (domain: MovieInformation, range: xsd:string)
:hasRuntime (domain: MovieInformation, range: xsd:float) - FUNCTIONAL - in minutes

### MovieResult Properties
:hasVoteAverage (domain: MovieResult, range: xsd:float) - FUNCTIONAL - scale 0-10
:hasVoteCount (domain: MovieResult, range: xsd:integer) - FUNCTIONAL
:hasRevenue (domain: MovieResult, range: xsd:float) - FUNCTIONAL - in USD

### MovieProduction Properties
:hasDirector (domain: MovieProduction, range: Director) - subPropertyOf hasCrewMember
:hasWriter (domain: MovieProduction, range: Writer) - subPropertyOf hasCrewMember
:hasCastMember (domain: MovieProduction, range: CastMember)
:hasCrewMember (domain: MovieProduction, range: CrewMember)
:hasProductionCompany (domain: MovieProduction, range: foaf:Organization)
:hasProductionCountry (domain: MovieProduction, range: xsd:string) - ISO 3166-1

### MovieContent Properties
:hasGenre (domain: MovieContent, range: Genre)
:hasKeyword (domain: MovieContent, range: Keyword)
:hasCollection (domain: MovieContent, range: Collection)
:hasPlot (domain: MovieContent, range: xsd:string)
:collectionName (domain: Collection, range: xsd:string)

### Person Properties
:personName (domain: Person, range: xsd:string)
:personGender (domain: Person, range: xsd:string)
:job (domain: CrewMember, range: xsd:string)

### CastMember & Character Properties
:playsCharacter (domain: CastMember, range: Character) - inverse: performedBy
:performedBy (domain: CastMember, range: Person) - FUNCTIONAL
:characterName (domain: Character, range: xsd:string)
:castOrder (domain: CastMember, range: xsd:integer) - billing order

### Rating Properties
:hasValue (domain: Rating, range: xsd:float) - FUNCTIONAL - 0.5 to 5.0 by 0.5 steps
:hasUserId (domain: User, range: xsd:integer) - FUNCTIONAL
:givenBy (domain: Rating, range: User) - inverse: gaveRating
:ratedMovie (domain: User, range: Movie)

### Award Properties
:awardName (domain: Award, range: xsd:string) - FUNCTIONAL
:hasYearOfAwardGiving (domain: Award, range: xsd:gYear) - FUNCTIONAL
:awardCategory (domain: Award, range: xsd:string)
:awardedBy (domain: Award, range: foaf:Organization)
:receivedBy (domain: Award, range: Person) - inverse: wonAward
:isNomination (domain: Award, range: xsd:boolean) - FUNCTIONAL

### Other Properties
:songTitle (domain: Song, range: xsd:string) - FUNCTIONAL
:hasComposer (domain: Song, range: Composer)
:adaptationYear (domain: FrenchAdaptation, range: xsd:gYear) - FUNCTIONAL
:hasAdaptor (domain: FrenchAdaptation, range: Person)
:hasFrenchVoiceActor (domain: FrenchAdaptation, range: VoiceActor)
:dubsCharacter (domain: VoiceActor, range: Character)

## INFERRED PROPERTIES (Property Chain Axioms)

:directedBy (Movie -> Director)
  INFERRED VIA: :hasProduction / :hasDirector
  
:writtenBy (Movie -> Writer)
  INFERRED VIA: :hasProduction / :hasWriter

:hasGenreOf (Movie -> Genre)
  INFERRED VIA: :hasContent / :hasGenre

:hasComposerOf (Movie -> Composer)
  INFERRED VIA: :hasThemeSong / :hasComposer

:actedIn (CastMember -> Movie)
  INVERSE OF: :hasCastMember

:sameCollection (Movie -> Movie) - SYMMETRIC, TRANSITIVE
  Movies in same franchise

:coDirectedWith (Person -> Person) - SYMMETRIC
  Directors who co-directed a movie

:coActedWith (Person -> Person) - SYMMETRIC
  Actors who acted together

:wonAward (Person -> Award)
  INVERSE OF: :receivedBy

## IMPORTANT QUERY PATTERNS

To find a movie's director:
  ?movie :hasTitle "MovieName" ;
         :hasProduction ?prod .
  ?prod :hasDirector ?director .
  ?director :personName ?name .

OR using inferred property (if reasoner is active):
  ?movie :hasTitle "MovieName" ;
         :directedBy ?director .
  ?director :personName ?name .

To find movie release date:
  ?movie :hasTitle "MovieName" ;
         :hasInformation ?info .
  ?info :hasReleaseDate ?date .

To find movies by genre:
  ?movie :hasContent ?content .
  ?content :hasGenre genres:Action .

To find actors in a movie:
  ?movie :hasTitle "MovieName" ;
         :hasProduction ?prod .
  ?prod :hasCastMember ?actor .
  ?actor :personName ?name .

To count movies by genre:
  SELECT (COUNT(?movie) AS ?count) WHERE {
    ?movie :hasContent ?content .
    ?content :hasGenre genres:Action .
  }
"""

SPARQL_TEMPLATE = """
You are a SPARQL query generator for a Movie Knowledge Graph.

YOUR TASK: Generate a SPARQL query to answer this question:
{question}

ONTOLOGY SCHEMA:
{schema}

REQUIREMENTS:
1. MUST use PREFIX : <http://saraaymericradhi.org/movie-ontology#>
2. MUST use PREFIX genres: <http://saraaymericradhi.org/movie-ontology/genres#> when querying genres
3. The ontology uses a MODULAR structure:
   - Movie -> :hasInformation -> MovieInformation (properties: :hasBudget, :hasReleaseDate, :hasRuntime, :hasLanguage)
   - Movie -> :hasResult -> MovieResult (properties: :hasVoteAverage, :hasVoteCount, :hasRevenue)
   - Movie -> :hasProduction -> MovieProduction (properties: :hasDirector, :hasCastMember, :hasProductionCompany)
   - Movie -> :hasContent -> MovieContent (properties: :hasGenre, :hasKeyword, :hasPlot)
4. Return ONLY the SPARQL query inside ```sparql ... ``` code block
5. NO explanations or additional text

REFERENCE EXAMPLES (DO NOT COPY - GENERATE A NEW QUERY FOR THE QUESTION ABOVE):

Example 1 - Budget query:
```sparql
PREFIX : <http://saraaymericradhi.org/movie-ontology#>
SELECT ?budget WHERE {{
  ?movie :hasTitle "Thor" ;
         :hasInformation ?info .
  ?info :hasBudget ?budget .
}}
```

Example 2 - Director query:
```sparql
PREFIX : <http://saraaymericradhi.org/movie-ontology#>
SELECT ?directorName WHERE {{
  ?movie :hasTitle "Iron Man 2" ;
         :hasProduction ?prod .
  ?prod :hasDirector ?director .
  ?director :personName ?directorName .
}}
```

Example 3 - Genre query:
```sparql
PREFIX : <http://saraaymericradhi.org/movie-ontology#>
PREFIX genres: <http://saraaymericradhi.org/movie-ontology/genres#>
SELECT ?title WHERE {{
  ?movie :hasTitle ?title ;
         :hasContent ?content .
  ?content :hasGenre genres:Action .
}} LIMIT 10
```

REMEMBER: Generate a NEW query for the question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=SPARQL_TEMPLATE
)

def extract_sparql(llm_output: str) -> str:
    """Extracts SPARQL code from LLM response."""
    match = re.search(r"```sparql\n(.*?)\n```", llm_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try finding just the SELECT ...
    if "SELECT" in llm_output:
        return llm_output.strip()
    return ""

def execute_sparql(graph: rdflib.Graph, query: str):
    """Executes SPARQL query on local graph."""
    try:
        results = graph.query(query)
        return list(results)
    except Exception as e:
        return f"Error: {e}"

def format_results(results: List[Any], query_type="SELECT") -> str:
    if isinstance(results, str): # Error message
        return results
    
    if not results:
        return "No results found."
    
    formatted = []
    for row in results:
        # row is a rdflib.query.ResultRow
        formatted.append(str(row))
    return "\n".join(formatted)

def sparql_pipeline(question: str, graph: rdflib.Graph):
    """Full Text-to-SPARQL pipeline."""
    print(f"Generating SPARQL for: '{question}'...")
    chain = PROMPT | LLM
    response = chain.invoke({"schema": ONTOLOGY_SCHEMA, "question": question})
    sparql_query = extract_sparql(response.content)
    
    print(f"Generated SPARQL:\n{sparql_query}")
    
    if not sparql_query:
        return "Failed to generate SPARQL query."
    
    results = execute_sparql(graph, sparql_query)
    formatted_answer = format_results(results)
    return formatted_answer

if __name__ == "__main__":
    # Test run
    g = load_graph()
    q = "What is the release date of 'Iron Man 2'?"
    ans = sparql_pipeline(q, g)
    print("Answer:", ans)
