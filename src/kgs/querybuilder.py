import src.utils.constants as const

def get_query_per_answer_type(answer_type):
    if answer_type == const.ANS_TYPE_DATE:
        return get_with_qualifier
    elif answer_type == const.ANS_TYPE_NUMBER:
        return get_numerical_lit
    elif answer_type == const.ANS_TYPE_OTHER:
        return get_query_so_hops
    else:
        raise ValueError(f"Unknown answer type: {answer_type}")

def get_with_qualifier(entity, qualifier_value, hops=1, prop_dict=None):
    """
    test query:
    SELECT ?p ?p1 ?pLabel ?o ?oLabel ?qv ?statement ?statementLabel ?p1Label WHERE {
        wd:Q171310 ?p ?statement .  # P166 = award received
        ?statement ?p1 ?qv .  # Get the point-in-time qualifier

        FILTER(CONTAINS(STR(?qv), "1988"))  # Ensure the qualifier contains "1962"
        
        # Fetch labels
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
        }
    }
    
    """
    select, where_clauses = get_statement_decoding_(entity, hops=hops)
    select_formatted = "SELECT " + " ".join([i for i in select])
    where_clause_formatted = " WHERE {    \n" + "    ".join(where_clauses) + ""
    where_clause_formatted = where_clause_formatted.replace(".", ".\n")
    
    # property ignore list
    timed_props = const.get()
    timed_props_formatted = "{" + " ".join(["pq:"+i for i in prop_dict.keys()]) + "}"
    
    filter_props_string = f"VALUES ?p3 {timed_props_formatted}" # FILTER (!REGEX(STR(?property), "P1477|1813"))
    # add qualifier part
    where_clause_formatted += f"FILTER(CONTAINS(STR({where_clauses[-1][:-1]}), \'{qualifier_value}\'))"  # Ensure the qualifier contains \"1962\"\n"
    
    filters = """
        FILTER CONTAINS(str(?p1), 'wikidata.org/prop/direct/')
        SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }
    """
    final_query = f"{_define_prefixes()}" + \
                  f"{select_formatted}" + \
                  f"{where_clause_formatted}" + \
                  f"{filters}" + \
                  f"{filter_props_string}"
    final_query += '}' # close it off
    return add_eos(final_query) # eos as custom protocol


def get_numerical_lit(entity, qualifier_value, hops=1, prop_dict=None):
    select, where_clauses = get_multihop_select_where_vars(entity, hops=hops)
    select_formatted = "SELECT " + " ".join([i for i in select])
    where_clause_formatted = " WHERE {    \n" + "    ".join(where_clauses) + ""
    where_clause_formatted = where_clause_formatted.replace(".", ".\n")
    
    # property ignore list
    timed_props_formatted = "{" + " ".join(["wdt:"+i for i in prop_dict.keys()]) + "}"
    
    filter_props_string = f"VALUES ?p3 {timed_props_formatted}" # FILTER (!REGEX(STR(?property), "P1477|1813"))
    # add qualifier part
    where_clause_formatted += f"FILTER(CONTAINS(STR({where_clauses[-1][:-1]}), \'{qualifier_value}\'))"  # Ensure the qualifier contains \"1962\"\n"
    
    filters = """
        FILTER CONTAINS(str(?p1), 'wikidata.org/prop/direct/')
        SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }
    """
    final_query = f"{_define_prefixes()}" + \
                  f"{select_formatted}" + \
                  f"{where_clause_formatted}" + \
                  f"{filters}" + \
                  f"{filter_props_string}"
    final_query += '}' # close it off
    return add_eos(final_query) # eos as custom protocol

    

    
def get_timed_properties(sparql_vars, list_of_props):
    output = []
    for i in sparql_vars:
        output.append((i, "|".join(list_of_props)))
    return output
    
def get_ignore_properties(sparql_prop_vars, list_of_props_to_ignore):
    output = []
    for i in sparql_prop_vars:
        output.append((i, "|".join(list_of_props_to_ignore)))
    return output


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

    select_vars, where_clauses = get_multihop_select_where_vars(subject, obj, hops)
    
    # create raw variables + labels
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


def get_label_of_entity(entity):
    q = f"""
        {_define_prefixes()}
        select * where {{
            wd:{entity} rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
        }} 
        LIMIT 1
        """
    return add_eos(q)

def get_entity_metadata(entity):
    q = f"""
        {_define_prefixes()}
        select ?l ?i ?s ?lLabel ?iLabel ?sLabel where {{
            wd:{entity} rdfs:label ?l .
            
            OPTIONAL {{
                wd:{entity} wdt:P31 ?i .
            }}
            
            OPTIONAL {{
                wd:{entity} wdt:P279 ?s .  
            }}
            
            FILTER (langMatches( lang(?l), "EN" ) )

            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 1
    """
    return add_eos(q)



def get_object(entity, property):
    q = f"""
        {_define_prefixes()}
        
        SELECT ?inst ?instLabel ?subCl ?subClLabel WHERE {{
          wd:{entity} wdt:P31 ?inst .
          wd:{entity} wdt:P279 ?subCl .


        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
    """
    
    # cleanup the string
    return add_eos(q)

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

def get_dbpedia_sameas_query(entity):
    q =  f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dbp: <http://dbpedia.org/property/>

        SELECT ?wikidataEntity WHERE {{
            dbr:{entity} owl:sameAs ?wikidataEntity .
            FILTER (CONTAINS(STR(?wikidataEntity), "wikidata.org"))
        }}
    """
    return q
    
    

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

def get_statement_decoding_(subj, hops=1):
    """This returns a query for retrieving a qualifier (e.g.) date of a target entity. The idea is to use this if we know
    the qualifier value. This would return e.g.
    SELECT DISTINCT ?p1 ?o1 ?p2 ?o2 ?pStatement ?when WHERE {
        wd:Q106624 ?p1 ?o1 .    # … with an awarded(P166) statement (o1 == e.g. wdt:Q2231-3FASFAS-AFRASR)
        ?o1 ?p2 ?o2 .           # decode the statement target like this (o2 == e.g. Nobel Prize)
        ?o1 ?pStatement ?when .   
    }

    Args:
        subj (_type_): _description_
        hops (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    select_vars = [f"?p1", f"?o1"]
    where_clauses = [f"wd:{subj}", f"?p1", f"?o1 ."] # always start with subject
    
    # standard hops
    for i in range(2, hops + 2): # we always need an extra hop
        select_vars.append(f"?p{i}")
        select_vars.append(f"?o{i}")
        
        where_clauses.append(f"?o{i - 1}")
        where_clauses.append(f"?p{i}")
        where_clauses.append(f"?o{i} .")

    # add the statement decoding
    obj_begin = where_clauses[-4].replace(" .", "") # 2nd last object
    p_statement = "?pStatement"
    o_statement = "?when ."
    
    select_vars.append(p_statement)
    select_vars.append(o_statement)
    where_clauses.append(obj_begin)
    where_clauses.append(p_statement)
    where_clauses.append(o_statement)
    
    return select_vars, where_clauses


def get_multihop_select_where_vars(subj, obj=None, hops=1):
    select_vars = [f"?p1", f"?o1"]
    where_clauses = [f"wd:{subj}", f"?p1", f"?o1 ."] # always start with subject
    

    for i in range(2, hops + 1):
        select_vars.append(f"?p{i}")
        select_vars.append(f"?o{i}")
        
        where_clauses.append(f"?o{i - 1}")
        where_clauses.append(f"?p{i}")
        where_clauses.append(f"?o{i} .")

    # always replace the last object with the target
    if obj is not None:
        select_vars.pop(-1)
        where_clauses[-1] = f"wd:{obj} ."
    
    return select_vars, where_clauses