def _define_prefixes():
    return r"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX wdv: <http://www.wikidata.org/value/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX dct: <http://purl.org/dc/terms/>
    """
    
def add_eos(query):
    return query + "<|EOS|>"

def get_query_so_hops(subject, obj, hops):
    prefixes = _define_prefixes()

    select_vars = [f"?p1", f"?o1"]
    where_clauses = [f"wd:{subject}", f"?p1", f"?o1 ."] # always start with subject
    

    for i in range(2, hops + 1):
        select_vars.append(f"?p{i}")
        select_vars.append(f"?o{i}")
        
        where_clauses.append(f"?o{i - 1}")
        where_clauses.append(f"?p{i}")
        where_clauses.append(f"?o{i} .")

    # always replace the last object with the target
    select_vars.pop(-1)
    where_clauses[-1] = f"wd:{obj} ."
    
    select_clause = "SELECT " + " ".join([i for i in select_vars])
    select_clause += " " + " ".join([f"{i}Label" for i in select_vars])
    where_clause = " WHERE {    \n" + "    ".join(where_clauses) + ""
    where_clause = where_clause.replace(".", ".\n")
    filters = """
        FILTER CONTAINS(str(?p1), 'wikidata.org/prop/direct/')
        SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }
    """

    final_query = f"{prefixes}" + \
                  f"{select_clause}" + \
                  f"{where_clause}" + \
                  f"{filters}"
                    
    final_query += '}' # close it off
    
    return add_eos(final_query) # eos as custom protocol 




def getquery(entity):
    q = f"""
        {_define_prefixes()}
        
        SELECT ?p ?o ?pLabel ?oLabel WHERE {{
            wd:{entity} ?p ?o .
        FILTER CONTAINS(str(?p),"wikidata.org/prop/direct/")
        FILTER CONTAINS(str(?o),"en")

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
    """
    
    # cleanup the string
    return add_eos(q)

def getAKAquery(entity):
    q = f"""
        {_define_prefixes()}
        
        SELECT DISTINCT ?entity ?entityLabel ?alias
        WHERE {{
            # Search for the entity using its alias or label
            ?entity ?labelOrAlias "{entity}"@en.
            VALUES ?labelOrAlias {{ rdfs:label skos:altLabel }}
            
            # Get the aliases
            OPTIONAL {{
                ?entity skos:altLabel ?alias.
                FILTER(LANG(?alias) = "en") # Restrict aliases to English
            }}
            
            # Get the label of the entity
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
    """
    
    # cleanup the string
    return add_eos(q)