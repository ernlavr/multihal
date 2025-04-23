import src.utils.constants as const
import src.utils.helpers as helpers

def get_query_per_answer_type(answer_type):
    if answer_type == const.ANS_TYPE_DATE:
        return lambda ent, qual_val, hops: get_with_qualifier(ent, qual_val, hops, const.get_list_of_timed_props(const.PROPS_TIMED_PATH))
    elif answer_type == const.ANS_TYPE_NUMBER:
        return lambda ent, qual_val, hops: get_numerical_lit(ent, qual_val, hops, const.get_list_of_timed_props(const.PROPS_NUM_PATH), add_unit=True)
    elif answer_type == const.ANS_TYPE_RANK:
        return lambda ent, qual_val, hops: get_numerical_lit(ent, qual_val, hops, const.get_list_of_rank_properties(), add_unit=False)
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
    
    qualifier_value = helpers.parse_flexible_date(qualifier_value)
    
    # property ignore list
    timed_props_formatted = "{" + " ".join([f"pq:{i} ps:{i}" for i in prop_dict.keys()]) + "}"
    
    filter_props_string = f"VALUES {where_clauses[-2]} {timed_props_formatted}" # FILTER (!REGEX(STR(?property), "P1477|1813"))
    # add qualifier part
    where_clause_formatted += f"FILTER(CONTAINS(STR({where_clauses[-1][:-1].strip()}), \'{qualifier_value}\'))"  # Ensure the qualifier contains \"1962\"\n"
    
    filters = " SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. }"
    final_query = f"{_define_prefixes()}" + \
                  f"{select_formatted}" + \
                  f"{where_clause_formatted}" + \
                  f"{filters}" + \
                  f"{filter_props_string}"
    final_query += '}' # close it off
    return add_eos(final_query) # eos as custom protocol


def get_numerical_lit(entity, qualifier_value, hops=1, prop_dict=None, add_unit=False):
    select, where_clauses = get_statement_decoding_(entity, hops=hops)
    if add_unit:
        select.append('?o99')
    select_formatted = "SELECT " + " ".join([i for i in select])
    where_clause_formatted = " WHERE {    \n" + "    ".join(where_clauses) + ""
    where_clause_formatted = where_clause_formatted.replace(".", ".\n")
    
    # Define allowed properties, this is the last property
    _timed_props_formatted = "{" + " ".join([f"pq:{k} ps:{k}" for k in prop_dict.keys()]) + "}"
    filter_props_string = f"VALUES {where_clauses[-2]} {_timed_props_formatted} \n" # FILTER (!REGEX(STR(?property), "P1477|1813"))
    
    # add filters
    filter_numeric = f"FILTER(isNumeric({where_clauses[-1][:-1]})) \n" # our target needs to be numeric
    where_clause_formatted += f"FILTER (STR({where_clauses[-1][:-1]}) = '{qualifier_value}') \n"  # Ensure the qualifier contains \"1962\"\n"    
    
    # add optional for deriving a unit for numeric
    label_service = "SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en'. } \n"
    
    final_query = f"{_define_prefixes()}" + \
                  f"{select_formatted}" + \
                  f"{where_clause_formatted}" + \
                  f"{filter_numeric}" + \
                  f"{label_service}" + \
                  f"{filter_props_string}"
    if add_unit:
        optional_unit = f"OPTIONAL {{ {where_clauses[-4][:-1]} wikibase:quantityUnit {select[-1]} . }} \n"
        final_query += f"{optional_unit}"
        
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
            FILTER (
                LANG(?label) = "en" || LANG(?label) = "mul"
            )
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

        SELECT ?wikidataEntity ?wikidataEntityLabel WHERE {{
            dbr:{entity} owl:sameAs ?wikidataEntity .
            FILTER (CONTAINS(STR(?wikidataEntity), "wikidata.org"))
            
            dbr:{entity} rdfs:label ?wikidataEntityLabel .
            FILTER (lang(?wikidataEntityLabel) = "en")
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
    for i in range(2, hops + 3): # we always need an extra two hops, first hop is is statement, then statement label, then qualifier
        select_vars.append(f"?p{i}")
        select_vars.append(f"?o{i}")
        
        where_clauses.append(f"?o{i - 1}")
        where_clauses.append(f"?p{i}")
        where_clauses.append(f"?o{i} .")

    # add the statement decoding
    where_clauses[-3] = where_clauses[-6]
    
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



# with_unit = """
# SELECT ?statement ?statementLabel ?val ?valuenode ?unit ?unitLabel WHERE {
#   wd:Q2412190 p:P2046 ?statement .  # Get the numeric value
#   ?statement psv:P2046 ?valuenode .
#   OPTIONAL { ?valuenode wikibase:quantityUnit ?unit . }  # Retrieve unit if available
#   OPTIONAL {}
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
# }
# """

with_union = """
SELECT ?p ?p1 ?pLabel ?o ?oLabel ?qv ?val ?statement ?statementLabel ?p1Label ?p2 ?o2 ?p3 ?unit WHERE {

  # CASE 1: literal
  {
      wd:Q76 ?p ?statement .  # get direct literal
      ?statement ?p1 ?val .
  }
#   UNION
#   {
#       wd:Q76 wdt:P26 ?statement .  # P166 = award received
#       ?statement ?p2 ?o2 .    # catch the label of the statement
#       ?statement ?p3 ?val .   # get the qualifier
#   }
  
  VALUES ?p1 {p:P111 ps:P111 p:P795 ps:P795 p:P873 ps:P873 p:P1117 ps:P1117 p:P1122 ps:P1122 p:P1123 ps:P1123 p:P1126 ps:P1126 p:P1127 ps:P1127 p:P1295 ps:P1295 p:P2043 ps:P2043 p:P2044 ps:P2044 p:P2046 ps:P2046 p:P2047 ps:P2047 p:P2048 ps:P2048 p:P2049 ps:P2049 p:P2050 ps:P2050 p:P2051 ps:P2051 p:P2052 ps:P2052 p:P2053 ps:P2053 p:P2054 ps:P2054 p:P2055 ps:P2055 p:P2056 ps:P2056 p:P2060 ps:P2060 p:P2066 ps:P2066 p:P2067 ps:P2067 p:P2068 ps:P2068 p:P2069 ps:P2069 p:P2073 ps:P2073 p:P2075 ps:P2075 p:P2076 ps:P2076 p:P2077 ps:P2077 p:P2097 ps:P2097 p:P2101 ps:P2101 p:P2102 ps:P2102 p:P2107 ps:P2107 p:P2112 ps:P2112 p:P2113 ps:P2113 p:P2114 ps:P2114 p:P2116 ps:P2116 p:P2117 ps:P2117 p:P2118 ps:P2118 p:P2119 ps:P2119 p:P2120 ps:P2120 p:P2128 ps:P2128 p:P2129 ps:P2129 p:P2144 ps:P2144 p:P2145 ps:P2145 p:P2148 ps:P2148 p:P2149 ps:P2149 p:P2150 ps:P2150 p:P2151 ps:P2151 p:P2154 ps:P2154 p:P2160 ps:P2160 p:P2177 ps:P2177 p:P2199 ps:P2199 p:P2202 ps:P2202 p:P2211 ps:P2211 p:P2216 ps:P2216 p:P2217 ps:P2217 p:P2227 ps:P2227 p:P2228 ps:P2228 p:P2229 ps:P2229 p:P2230 ps:P2230 p:P2231 ps:P2231 p:P2233 ps:P2233 p:P2234 ps:P2234 p:P2248 ps:P2248 p:P2250 ps:P2250 p:P2254 ps:P2254 p:P2260 ps:P2260 p:P2261 ps:P2261 p:P2262 ps:P2262 p:P2300 ps:P2300 p:P2362 ps:P2362 p:P2370 ps:P2370 p:P2386 ps:P2386 p:P2404 ps:P2404 p:P2430 ps:P2430 p:P2436 ps:P2436 p:P2442 ps:P2442 p:P2527 ps:P2527 p:P2528 ps:P2528 p:P2532 ps:P2532 p:P2542 ps:P2542 p:P2547 ps:P2547 p:P2556 ps:P2556 p:P2557 ps:P2557 p:P2565 ps:P2565 p:P2583 ps:P2583 p:P2645 ps:P2645 p:P2659 ps:P2659 p:P2710 ps:P2710 p:P2717 ps:P2717 p:P2718 ps:P2718 p:P2781 ps:P2781 p:P2784 ps:P2784 p:P2791 ps:P2791 p:P2793 ps:P2793 p:P2797 ps:P2797 p:P2806 ps:P2806 p:P2807 ps:P2807 p:P2808 ps:P2808 p:P2873 ps:P2873 p:P2911 ps:P2911 p:P2923 ps:P2923 p:P2957 ps:P2957 p:P3013 ps:P3013 p:P3020 ps:P3020 p:P3039 ps:P3039 p:P3041 ps:P3041 p:P3070 ps:P3070 p:P3071 ps:P3071 p:P3078 ps:P3078 p:P3157 ps:P3157 p:P3251 ps:P3251 p:P3252 ps:P3252 p:P3253 ps:P3253 p:P4036 ps:P4036 p:P4163 ps:P4163 p:P4250 ps:P4250 p:P4296 ps:P4296 p:P4511 ps:P4511 p:P5066 ps:P5066 p:P5067 ps:P5067 p:P5141 ps:P5141 p:P5608 ps:P5608 p:P5670 ps:P5670 p:P5672 ps:P5672 p:P5673 ps:P5673 p:P5674 ps:P5674 p:P5675 ps:P5675 p:P5676 ps:P5676 p:P5677 ps:P5677 p:P5678 ps:P5678 p:P5679 ps:P5679 p:P5681 ps:P5681 p:P5682 ps:P5682 p:P5685 ps:P5685 p:P5706 ps:P5706 p:P5708 ps:P5708 p:P5709 ps:P5709 p:P5993 ps:P5993 p:P6014 ps:P6014 p:P6272 ps:P6272 p:P6710 ps:P6710 p:P6856 ps:P6856 p:P6876 ps:P6876 p:P7015 ps:P7015 p:P7297 ps:P7297 p:P8111 ps:P8111 p:P8497 ps:P8497 p:P8597 ps:P8597 p:P8628 ps:P8628 p:P9059 ps:P9059 p:P9998 ps:P9998 p:P10107 ps:P10107 p:P12004 ps:P12004 p:P12571 ps:P12571 p:P1198 ps:P1198 p:P1279 ps:P1279 p:P1689 ps:P1689 p:P2595 ps:P2595 p:P2661 ps:P2661 p:P2663 ps:P2663 p:P2665 ps:P2665 p:P2834 ps:P2834 p:P2855 ps:P2855 p:P2927 ps:P2927 p:P5594 ps:P5594 p:P5811 ps:P5811 p:P5893 ps:P5893 p:P5895 ps:P5895 p:P5896 ps:P5896 p:P5898 ps:P5898 p:P5929 ps:P5929 p:P6076 ps:P6076 p:P6639 ps:P6639 p:P6897 ps:P6897 p:P7079 ps:P7079 p:P1113 ps:P1113 p:P1114 ps:P1114 p:P1436 ps:P1436 p:P2130 ps:P2130 p:P2137 ps:P2137 p:P2138 ps:P2138 p:P2139 ps:P2139 p:P2212 ps:P2212 p:P2218 ps:P2218 p:P2240 ps:P2240 p:P2284 ps:P2284 p:P2295 ps:P2295 p:P2437 ps:P2437 p:P2555 ps:P2555 p:P2599 ps:P2599 p:P2635 ps:P2635 p:P2660 ps:P2660 p:P2664 ps:P2664 p:P2712 ps:P2712 p:P2769 ps:P2769 p:P2803 ps:P2803 p:P2896 ps:P2896 p:P2929 ps:P2929 p:P3036 ps:P3036 p:P3063 ps:P3063 p:P3086 ps:P3086 p:P3264 ps:P3264 p:P3487 ps:P3487 p:P3575 ps:P3575 p:P3737 ps:P3737 p:P3738 ps:P3738 p:P3740 ps:P3740 p:P4131 ps:P4131 p:P4214 ps:P4214 p:P4268 ps:P4268 p:P4269 ps:P4269 p:P4519 ps:P4519 p:P4876 ps:P4876 p:P4895 ps:P4895 p:P5043 ps:P5043 p:P5045 ps:P5045 p:P5065 ps:P5065 p:P5348 ps:P5348 p:P5524 ps:P5524 p:P5582 ps:P5582 p:P5822 ps:P5822 p:P5899 ps:P5899 p:P6753 ps:P6753 p:P7328 ps:P7328 p:P7584 ps:P7584 p:P7862 ps:P7862 p:P8093 ps:P8093 p:P8393 ps:P8393 p:P8757 ps:P8757 p:P9180 ps:P9180 p:P9927 ps:P9927 p:P10209 ps:P10209 p:P10263 ps:P10263 p:P10322 ps:P10322 p:P10648 ps:P10648 p:P11698 ps:P11698 p:P12469 ps:P12469 p:P12470 ps:P12470 p:P12471 ps:P12471 p:P12549 ps:P12549 p:P12651 ps:P12651 p:P13171 ps:P13171 p:P1111 ps:P1111 p:P1697 ps:P1697 p:P5044 ps:P5044 p:P1082 ps:P1082 p:P1083 ps:P1083 p:P1098 ps:P1098 p:P1110 ps:P1110 p:P1120 ps:P1120 p:P1128 ps:P1128 p:P1132 ps:P1132 p:P1174 ps:P1174 p:P1339 ps:P1339 p:P1342 ps:P1342 p:P1345 ps:P1345 p:P1373 ps:P1373 p:P1410 ps:P1410 p:P1446 ps:P1446 p:P1539 ps:P1539 p:P1540 ps:P1540 p:P1561 ps:P1561 p:P1590 ps:P1590 p:P1831 ps:P1831 p:P1833 ps:P1833 p:P1867 ps:P1867 p:P1971 ps:P1971 p:P2124 ps:P2124 p:P2196 ps:P2196 p:P2573 ps:P2573 p:P3744 ps:P3744 p:P3872 ps:P3872 p:P4295 ps:P4295 p:P4909 ps:P4909 p:P5436 ps:P5436 p:P5630 ps:P5630 p:P6125 ps:P6125 p:P6343 ps:P6343 p:P6344 ps:P6344 p:P6498 ps:P6498 p:P6499 ps:P6499 p:P8687 ps:P8687 p:P9077 ps:P9077 p:P9107 ps:P9107 p:P9740 ps:P9740 p:P9924 ps:P9924 p:P10610 ps:P10610 p:P10623 ps:P10623 p:P12712 ps:P12712}
  
#   FILTER(datatype(?val) = xsd:decimal || 
#          datatype(?val) = xsd:double || 
#          datatype(?val) = xsd:integer)
  
  OPTIONAL { ?val wikibase:quantityUnit ?unit . }
  FILTER(isNumeric(?val))
  FILTER(CONTAINS(STR(?val), "1.85"))  # Ensure the qualifier contains "1962"
  #VALUES ?statement { "1961-08-04T00:00:00Z"^^xsd:date }  # Replace with your target date
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
  }
}
"""

# Query cases
# Direct literal (numeric, data)