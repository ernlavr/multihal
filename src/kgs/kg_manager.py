import spacy

import src.kgs.querybuilder as qb
import src.network.api.local_api as api
import src.network.api.endpoints as ep
import src.network.udp_manager as br
import src.utils.helpers as uti
import src.utils.logger as log
import src.utils.config as config
import src.analysis.figures as figs
import src.utils.constants as const
import src.utils.helpers as helpers

import time
import requests
import json
import polars as pl
import numpy as np
import re
import logging
from itertools import chain
from collections import defaultdict

from tqdm import tqdm
import re

class KGManager():

    def __init__(self, dataframe: pl.DataFrame, args, continue_from=None):
        self.logger = log.KgLogger(args, create_log=True, continue_from=continue_from)
        self.args = args
        self.bridge = br.NetworkBridge()
        self.dataframe = dataframe
        self.spacy_model = spacy.load("en_core_web_trf")
        self.api = api.WikidataAPI()
        self.query_engine = br.NetworkBridge()
        self.hops = 2
        self.dbpedia_cache = {}
        
        # Logging
        self.ignore_properties = uti.fill_ignore_properties("res/wd_properties_to_ignore/ids_to_remove.json")
        self.all_properties = uti.fill_all_properties('res/wd_properties_to_ignore/props_v2.json')

        # Matches <QXXX> where XXX is arbitray amount of numbers
        self.obj_regex = lambda x : bool(re.match(r"^Q\d+$", x))

        if continue_from:
            # load the dataframe
            self.df = pl.read_csv(f'results/{continue_from}.csv')

            # get last question ID
            last_qid = self.df['q_ID'].iloc[-1]
            # remove all questions from the data that are already processed
            for i, d in enumerate(self.data):
                if d['questionid'] == last_qid:
                    self.data = self.data.select(range(i+1, len(self.data)))
                    break

    def get_triple(self, subject, predicate, obj, qualifier=None):
        # subject, predicate, object are all dictionaries, merge them together
        trip = {**subject, **predicate, **obj}
        
        if qualifier:
            pass #TODO implement

        return trip
    
    
    def filter_paths(self, dataset: pl.DataFrame):
        """ Cleanup paths by removing paths which contain irrelevant properties,
        e.g. identifiers, URLs, images, audio references..
        """
        irrelevant_props = uti.fill_ignore_properties("res/wd_properties_to_ignore/ids_to_remove_V2.json")
        _RE_COMBINE_WHITESPACE = re.compile(r"\s+") # for cleaning up whitespaces
        
        total_paths = 0
        after_filtering = 0
        dataset = dataset.filter(~dataset['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        
        for i, row in tqdm(enumerate(dataset.iter_rows(named=True)), "Filtering paths", total=dataset.shape[0]):
            trips = row['responses'].split(config.LIST_SEP)
            if trips is None or len(trips) == 0:
                continue
                    
            # add to total paths
            total_paths += len(trips)
        
            # unify any whitespace noise, remove duplicate paths
            trips = [_RE_COMBINE_WHITESPACE.sub(" ", t).strip() for t in trips]
            trips = list(set(trips))  # Remove duplicates
            
            new_paths = []
            for trip in trips:
                # check if path contains any of the irrelevant properties
                skip = False
                for t in trip.split():
                    if t in irrelevant_props:
                        skip = True
                        break
                        
                if skip is False:
                    new_paths.append(trip)
                
            
            after_filtering += len(new_paths)
            # update the row with the new paths
            
            
            new_paths = f"{config.LIST_SEP}".join(new_paths)
            row['responses'] = new_paths
            
            _datapoint = pl.from_dict(row, strict=False)
            dataset = dataset.update(_datapoint, on="id")
        
        logging.info(f"Total paths: {total_paths}; After filtering: {after_filtering}")
        return dataset
    
    def decode_statement_labels(self, statement: list, cache:dict, datapoint):
        labels = []
        
        literal = None
        if datapoint['answer_type'] != const.ANS_TYPE_OTHER:
            #statement.pop(-1) # dates/numericals have literal last position
            literal = datapoint['output']
        
        for entity in statement:
            if entity in cache:
                labels.append(cache[entity])
                continue
            
            # _statement = helpers.remove_duplicate_hops(statement)
            # if _statement != statement:
            #     logging.info(f"Removing duplicate hops: \n{statement} \n{_statement}")
            #     statement = _statement
            
            # statement box e.g. "Q76-fsa4a43-4a3-a234", entity a potential literal
            if helpers.is_entity_literal(entity):
                # if entity is the last element of the statement, add the literal!
                if statement[-1] == entity:
                    if literal is not None:
                        labels.append(literal)
                else:
                    labels.append(entity)
                continue
            
            elif helpers.is_entity_statement(entity):
                idx = statement.index(entity)
                # skip the statement to linkup the path to the statement label
                if statement[idx - 1] == statement[idx + 1]:
                    tmp = statement.copy()
                    statement.pop(idx - 1)
                    statement.pop(idx - 1) # the list gets re-arranged and all elements shift to left..
                    logging.info(f"Skipping statement: {tmp} -> {statement};")
                    return self.decode_statement_labels(statement, cache, datapoint)
            
            elif helpers.is_entity_object(entity):
                query = qb.get_label_of_entity(entity)
                response = self.bridge.send_message(message=query)
                if len(response['results']['bindings']) == 0:
                    continue
                labels.append(response['results']['bindings'][0]['label']['value'])
                cache[entity] = labels[-1]
                
            elif helpers.is_entity_property(entity):
                property = self.all_properties.get(entity, None)
                if property is not None:
                    property = property['label']
                else:   # soft error handling, skip this whole path. This is not expected to happen though.
                    logging.warning("Property not found in all_properties: %s" % entity)
                    return None, cache
                labels.append(property)
                cache[entity] = labels[-1]
            else:
                return None, cache
            
            
        
        if len(statement) != len(labels):
            logging.error(f"Length of statement: {len(statement)} does not match length of labels: {len(labels)}; statement: {statement}; labels: {labels}")
            
        # lastly, add the literal back, we dont want to derive a label from it
        return labels, cache
            
    
    def checkValidDatapoint(self, datapoint):
        """ Check if a datapoint contains ans and ans_str """
        try:
            if len(datapoint['output']) == 0:
                return False
            if datapoint['output'] == [None]:
                return False
            
            return True
        except Exception as e:
            logging.error(f"Datapoint: {datapoint['id']} does not contain output")
    
    def tuple_exists_by_text(self, lst, tup):
        # Flatten the nested list and tuple structures
        flat_lst = [entity.text 
                    for entities in lst 
                        for entity in entities
                    ]
        flat_tup = [eentity.text 
                    for eentities in tup 
                        for eentity in (eentities 
                                        if hasattr(eentities, '__iter__') 
                                        else [eentities])]

        # Check if any text in the flattened list exists in the flattened tuple
        return any(text in flat_tup for text in flat_lst)

    def add_labels(self, data: pl.DataFrame):
        _data = data.filter(~data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        # add "trip_labels" column
        _data = _data.with_columns(
                    trip_labels=pl.lit('N/A')
                )
        
        # # control type of answers
        # _data = _data.filter(pl.col('answer_type') == const.ANS_TYPE_NUMBER)
        
        cache = {}
        logging.info("Adding labels to triples")
        
        for datapoint in tqdm(_data.iter_rows(named=True), total=_data.shape[0]):
            trips = datapoint.get('responses').split(config.LIST_SEP)
            logging.info(f"Processing row {datapoint['id']} with triples (n={len(trips)})")
            
            labels = []
            
            for trip in trips.copy(): # operate on a copy so we can remove invalid ones
                if len(trip) == 0: continue
                trip = trip.strip()
                # for each triple, decode the identifiers to labels    
                _labels, cache = self.decode_statement_labels(trip.split(), cache, datapoint)
                if _labels is None:
                    trips.remove(trip)
                    continue
                _labels = "; ".join(_labels)
                labels.append(_labels)
            
            if len(labels) != len(trips):
                logging.error(f"Length of labels: {len(labels)} does not match length of trips: {len(trips)}; \nlabels: {labels}; \ntrips: {trips}")
            
            labels = f"{config.LIST_SEP}".join(labels)
            trips = f"{config.LIST_SEP}".join(trips)
            datapoint['responses'] = trips
            datapoint['trip_labels'] = labels
            _datapoint = pl.from_dict(datapoint, strict=False)
            _data = _data.update(_datapoint, on="id")
            
            # save
            save_path = f"{self.args.data_dir}/data_with_wd_labels.json"
            _data.write_json(save_path)
            
        logging.info(f"Trip labels saved to: {save_path}")    
        return _data
    
    def get_wikidata_from_wikipedia(self, wikipedia, attempts=3):
        # contexts will have the form "xxx.org/wiki/ID#any_other_further_shit"
        wikipedia_id = wikipedia.split('wiki/')[-1]
        wikipedia_id = wikipedia_id.split('#')[0]
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={wikipedia_id}&format=json"
        data = None
        attempts -= 1
        
        # Parse web requests
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
        except Exception as e:
            logging.info("Query '%s' failed. Attempts: %s" % (wikipedia, str(attempts)))
            logging.info("\t" + str(e))
            time.sleep(21) #to avoid limit of calls, sleep 60s
            if attempts>0:
                return self.get_wikidata_from_wikipedia(wikipedia, attempts)
            else:
                return None
        
        # Extract wikidata ID
        wikidatas = []
        if (query := data.get('query', None)) is not None:
            if (pages := query.get('pages', None)) is not None:
                for k, v in pages.items():
                    if (pageprops := v.get('pageprops', None)) is not None:
                        if (wikidata := pageprops.get('wikibase_item', None)) is not None:
                            # add wikidata.org for consistency... maybe we can remvoe this
                            wikidatas.append(f"http://www.wikidata.org/entity/{wikidata}")
        if attempts < 1:
            return wikidatas
        
        if 'wiki/' in wikipedia and len(wikidatas) == 0:
            logging.error(f"No wikidata found for {wikipedia}. Something went wrong?. Debug: {url}")
            wid_split = wikipedia_id.split('_')
            wid_first_cap = wid_split[0].capitalize() + "_" + "_".join(wid_split[1:])
            wid_cap = "_".join([i.capitalize() for i in wid_split])
            wid_upper = "_".join([i.upper() for i in wid_split])
            
            logging.error("Attempting again with first cap")
            first_cap = self.get_wikidata_from_wikipedia(wid_first_cap, attempts=1)
            if first_cap is None or len(first_cap): 
                logging.info(f"Got with first cap: {wid_first_cap}")
                return first_cap
            
            logging.error("Attempting again with all first-caps letters")
            capitalized = self.get_wikidata_from_wikipedia(wid_cap, attempts=1)
            if capitalized is None or len(capitalized): 
                logging.info(f"Got with capitalized: {wid_cap}")
                return capitalized
            
            logging.error("Attempting again with all-caps")
            upper = self.get_wikidata_from_wikipedia(wid_upper, attempts=1)
            if upper is None or len(upper): 
                logging.info(f"Got with all caps: {wid_upper}")
                return upper
            
        return wikidatas
            
        
        
    def get_falcon_query(self, query_text, api_mode='long', timeout=5):
        url = f"https://labs.tib.eu/falcon/falcon2/api?mode={api_mode}&k=3&db=1"
        headers = {"Content-Type": "application/json"}
        data = {"text": query_text}
        logging.info("Running Falcon API")
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        except:
            logging.info(f"Falcon API Timeout t={timeout}s: {query_text}")
            return [f"<TIMEOUT>{data['text']}</TIMEOUT>"]
        
        end_time = time.time()
        logging.info(f"Query time (s): {end_time - start_time:.2f}")

        # get response code
        if response.status_code != 200:
            logging.error(f"Failed to get response from Falcon API: {response.status_code} - {response.reason}")
            return None
        
        logging.info(f"Query sucessful: {query_text}")
        
        response = response.json()
        output = []

        # safe unpacking
        entities_dbpedia = response.get('entities_dbpedia') or []
        wikidata_ents_from_dbpedia = self.process_dbpedia_ents(entities_dbpedia)
        
        entities_wikidata = response.get('entities_wikidata') or []
        combined_entities = entities_wikidata + wikidata_ents_from_dbpedia
        if len(combined_entities) == 0:
            logging.info(f"No entities found in either Wikidata or Dbpedia for query: {query_text}")
            return ["<NO_ENTITIES_FOUND>"]

        for i in combined_entities:
            output.append(i['URI'])
        
        return output
    
    def process_dbpedia_ents(self, entities):
        if len(entities) == 0:
            return []
        
        output = []
        endpoint = ep.DBpediaEndpoint()
        for i in entities:
            entity = i['URI'].split('/')[-1]
            
            # few cleanup heuristics
            if '(' in entity and ')' in entity:
                entity = entity.replace('(', '\(').replace(')', '\)')
            if ',' in entity:
                entity = entity.replace(',', '\,')
            if '&' in entity:
                entity = entity.replace('&', '\&')
            if entity.endswith('.'):
                entity = entity[:-1] + '\.'
            if entity.startswith('.'):
                entity = '\.' + entity[1:]
            if '!' in entity:
                entity = entity.replace('!', '\!')
            if "\'" in entity:
                entity = entity.replace("\'", "\\\'")
            if "+" in entity:
                entity = entity.replace("+", "\+")
                
            entity = r"{}".format(entity)
            
            # try to get from cache
            cached = self.dbpedia_cache.get(entity, None)
            if cached is not None:
                for i in cached:
                    output.append({'URI': i})
                continue
            
            self.dbpedia_cache[entity] = []
            q = qb.get_dbpedia_sameas_query(entity)
            results = endpoint.getQueryResultsArityOne(q)
            if len(results) > 10:
                logging.info(f"DBPedia entity has {len(results)} results, skipping: {entity}")
                continue
            
            if results is not None:
                if len(results) > 1:
                    pass
                for i in results:
                    output.append({'URI': i})   
                    self.dbpedia_cache[entity].append(i)
        return output
                
            
    def process_queryable_entities(self, entity, entity_type, timeout=60):
        if 'N/A' in entity:
            query_result = self.get_falcon_query(entity_type, timeout=30, api_mode=self.args.api_mode)
            if query_result is None:
                logging.info(f"Error when querying Falcon API for {entity_type}, returned None.")
                return None
        
        elif f"<TIMEOUT>" in entity:
            logging.info(f"Processing timeout when querying Falcon API for {entity_type}.")
            # extract text between <TIMEOUT> </TIMEOUT>
            query_result = self.get_falcon_query(entity_type, api_mode=self.args.api_mode, timeout=timeout)
            if query_result is None:
                logging.info(f"Error when querying Falcon API for {entity_type}, returned None.")
                return None                
        else:
            return entity.split(config.LIST_SEP)
            
        # # Handle if a timeout happened in a list (e.g. objects)
        # if config.LIST_SEP in entity:
        #     entity = [i.strip(' ') for i in entity.split(config.LIST_SEP) if '<TIMEOUT>' not in i]
        #     entity.extend(query_result)
        # else:
        #     entity = query_result
        
        return list(set(query_result))  # Remove duplicates
    
    def process_falcon_strategy(self, datapoint):
        # Process queryable subjects
        queryable_subjects = self.process_queryable_entities(datapoint['subjects'], datapoint['input'])
        if queryable_subjects is None:
            return None
        
        # check if any of the entities are directly present in the datapoint
        
        queryable_subjects = uti.flatten_if_2d(queryable_subjects)
        queryable_subjects = list(set(queryable_subjects))  # Remove duplicates
        queryable_subjects = f" {config.LIST_SEP} ".join(queryable_subjects)
        
        # if the answer type is numeric or date, we dont need to query falcon, later we'll use those for special queries
        if datapoint['answer_type'] != const.ANS_TYPE_OTHER:
            queryable_objects = datapoint['output']
            return queryable_subjects, queryable_objects

        # Process queryable objects
        outputs = [datapoint['output']]
        if config.LIST_SEP in outputs:
            outputs = outputs.split(config.LIST_SEP)
            
        if datapoint['optional_output'] is not None:
            opt_outputs = datapoint['optional_output'].split(config.LIST_SEP)
            if len(opt_outputs) > 2:
                opt_outputs = opt_outputs[:2]
            outputs.extend(opt_outputs)
        
        outputs = list(set(outputs))  # Remove duplicates
        if len(outputs) > 5:
            outputs = outputs[:5]
            logging.info(f"Truncated outputs to 5 for datapoint: {datapoint['id']}")
        
        queryable_objects = []
        for output in outputs:
            qo = self.process_queryable_entities(datapoint['objects'], output)
            if qo is None:
                logging.info(f"Error when querying Falcon API for output ID {datapoint['id']}: {output}")
                continue
            queryable_objects.extend(qo)
        
        queryable_objects = uti.flatten_if_2d(queryable_objects)
        queryable_objects = list(set(queryable_objects))  # Remove duplicates
        queryable_objects = f" {config.LIST_SEP} ".join(queryable_objects)

        return (
            queryable_subjects,
            queryable_objects
        )
    
    def _get_kg_so_pairs(self, datapoint: dict, kg_name="wikidata", is_ans_other_type=False):
        """ Returns a list of tuples that has all pair-wise combinations between subject-object entities """
        # get the entities
        subjects = datapoint.get("subjects")
        objects = datapoint.get("objects")
        if subjects is None or objects is None:
            return []
        
        # Subjects/objects may be seperated by <SEP>
        subjects = subjects.split(config.LIST_SEP)
        objects = objects.split(config.LIST_SEP)
        subjects = list(set(subjects))
        objects = list(set(objects))

        # Finally get the pairs
        pairs = []
        s_equal_o = 0
        for s in subjects:
            if kg_name not in s:
                continue
            s = s.split('/')[-1].strip()

            for o in objects:
                if is_ans_other_type == False and kg_name in o:
                    continue
                
                if o is None or len(o) == 0:
                    continue
                
                # # extract only the QXXXX entity identifiers
                if kg_name in o:
                    o = o.split('/')[-1].strip()

                if 'dbpedia' in kg_name: # TODO cross-check this
                    raise NotImplementedError("DBPedia not implemented yet")
                
                # Prevent circular paths
                if s == o:
                    s_equal_o += 1
                    continue
                
                pairs.append((s, o))
                if is_ans_other_type:   # normally we wanna reverse it
                    pairs.append((o, s)) # reverse the pair

                
        
        logging.info(f"Number of pairs: {len(pairs)}")
        logging.info(f"Number of pairs where subject equals object: {s_equal_o}")
        return pairs, s_equal_o
    

    def parse_response(self, subj, obj, response: dict):
        """ Parse the response from the network bridge """
        paths = ""
        if response.get('error') is not None:
            logging.error(f"Error in response: {response.get('error')}") 
            return paths
        
        vars = response.get("results").get("bindings")
        if vars is None or len(vars) == 0:
            vars = []

            for i in response.get("results").get("bindings"):
                vars.append(i)

        else:
            for i in response['results']['bindings']:
                num_paths = len(response['results']['bindings'])
                filtered = [item for item in i.keys() if "Label" not in item]
                filtered.sort(key=lambda x: (int(x[1:]), x[0] == 'o'))
                path = ""
                for j in filtered:
                    val = i[j]['value']
                    
                    if 'Label' in j:
                        continue     
                    if 'o99' in j and 'Q199' in val:
                        continue
        
                    if 'rdf-syntax-ns' in val or 'prov#' in val or 'ontology#' in val or 'pqv:' in val:
                        path = ""
                        break              
                    else:
                        path += f"{i[j]['value']} ".split('/')[-1]
                
                if path == "":
                    continue
                
                # format path
                path = f"{subj} {path} "
                if obj is not None:
                    path += f"{obj} "
                path += f"{config.LIST_SEP}"
                
                paths += path
        
        
        return paths
    
    def get_queryable_datapoints(self, data: pl.DataFrame):
        """ Returns the datapoints that are queryable """
        return data.filter(
            (pl.col('subjects').str.contains("wikidata.org")) & 
            
                (~pl.col('objects').str.contains("TIMEOUT") & 
                ~pl.col('objects').str.contains("N/A") &
                ~pl.col('objects').str.contains("NO_ENTITIES_FOUND")
                ) &
            ((pl.col('responses') == 'N/A')))

    def cleanup_obj(self, obj: str):
        # remove start, end white space
        obj = obj.strip()
        # remove ending dots
        obj = obj.rstrip('.')
        # remove commas
        obj = obj.replace(',', '')
        return obj

    def query_kg(self, data: pl.DataFrame, network_bridge: br.NetworkBridge, max_hops=2, start_hop=1):
        if "responses" not in data.columns:
            data = data.with_columns(
                responses=pl.lit('N/A')
            )
        start_time = time.time()
        
        queryable_datapoints = self.get_queryable_datapoints(data)
        progress_bar = tqdm(queryable_datapoints.iter_rows(named=True), total=queryable_datapoints.shape[0])
        # datapoint add column responses
        
        total_number_of_pairs = 0
        total_number_of_circular = 0
        total_number_of_queries = 0
        for datapoint in progress_bar:
            logging.info(f"Processing Datapoint: {datapoint['id']}")
            progress_bar.set_description(f"Querying Wikidata")

            # pair-wise path search between subject-object entities
            ans_is_type_other = datapoint.get('answer_type') == const.ANS_TYPE_OTHER
            pairs, number_of_circular = self._get_kg_so_pairs(datapoint, is_ans_other_type=ans_is_type_other)            
            if not pairs:
                continue
            
            total_number_of_circular += number_of_circular
            total_number_of_pairs += len(pairs)

            responses = []
            for subj, obj in pairs:
                curr_hop = start_hop
                obj = self.cleanup_obj(obj)
                if obj == "":
                    continue
                
                while curr_hop <= max_hops:
                    # plug in the pair in the query
                    total_number_of_queries += 1
                    query_time = time.time()
                    query_func = qb.get_query_per_answer_type(datapoint['answer_type'])
                    query = query_func(subj, obj, hops=curr_hop)
                    
                    # run the query and save the response
                    response = network_bridge.send_message(message=query, use_only_wd=True)
                    
                    query_time = time.time() - query_time
                    logging.info(f"Processed {datapoint['id']} query for subj ({subj}) obj ({obj}) in time {query_time:.3f}: \n; hop: {curr_hop}")
                    if response:
                        # semi-hacky way to check if we need to add obj.. for nums/dates the object will be last ?o variable in the query
                        parsed = self.parse_response(subj, obj, response) if ans_is_type_other else self.parse_response(subj, None, response)
                        if len(parsed) > 0:
                            responses.append(parsed)
                            break
                    # increase hop
                    curr_hop += 1
                    
            
            responses = " ".join(responses).strip()
            # remove last <SEP>
            if responses.endswith(config.LIST_SEP):
                responses = responses[:-len(config.LIST_SEP)]
            if len(responses) == 0:
                responses = "<NO_PATHS_FOUND>"

            # add responses to the dataframe
            datapoint['responses'] = responses
            _datapoint = pl.from_dict(datapoint, strict=False)
            data = data.update(_datapoint, on="id")
            data.write_json(f"{self.args.data_dir}/data_subj_obj_parsed_queried_wd.json")

        end_time = time.time()
        logging.info(f"Total time taken: {(end_time - start_time):.2f}s")
        logging.info(f"Total number of pairs: {total_number_of_pairs}")
        logging.info(f"Total number of circular pairs: {total_number_of_circular}")
        logging.info(f"Total number of pairs: {total_number_of_pairs}")
        logging.info(f"Total number of queries: {total_number_of_queries} (max hops: {max_hops})")
        logging.info(f"Number of queryable datapoints: {queryable_datapoints.shape[0]}")
        
        return data
            
            
    def get_unprocessed_datapoints(self, data: pl.DataFrame):
        """Returns unprocessed datapoints where either 'subjects' or 'objects' 
        is 'N/A' or contains the '<TIMEOUT>' tag.
        """
        output = data.filter(
            (pl.col('subjects').str.contains('N/A')) | 
            (pl.col('objects').str.contains('N/A')) | 
            (pl.col('subjects').str.contains("<TIMEOUT>")) | 
            (pl.col('objects').str.contains("<TIMEOUT>"))
        )
        return output

                
    def process(self, data: pl.DataFrame) -> dict:
        logging.info("<--- End of Header --->\n")
        self.df = self.logger.getResultDataframe(data, self.hops)
        start_time = time.time()
        
        # DEBUG: remove rows where answer_type is not "other"
        # data = data.filter(pl.col('answer_type') != const.ANS_TYPE_OTHER)
        
        data.write_json(f"deleteme_before_full.json")
        column_order = data['id'].to_list()
        
        # create two new columns for subjects and objects
        if 'subjects' not in data.columns and 'objects' not in data.columns:
            data = data.with_columns(
                subjects = pl.lit('N/A'),
                objects = pl.lit('N/A')
            )
        
        # get rows where either subjects or objects is N/A or <TIMEOUT>
        # replace subjects, objects "" with N/A
        def replace_cols(data, col, val, replace_val):
            return data.with_columns(pl.when(pl.col(col) == val).then(pl.lit(replace_val)).otherwise(pl.col(col)).alias(col))
        data = replace_cols(data, "subjects", "", "N/A")
        data = replace_cols(data, "objects", "", "N/A")
        
        data_to_query = self.get_unprocessed_datapoints(data)
        data_to_not_query = data.join(data_to_query, on="id", how="anti")
        num_wikis_from_context = 0
        
        for idx, datapoint in enumerate(tqdm(data_to_query.iter_rows(named=True), "Processing Dataset", total=data_to_query.shape[0])):
            if not self.checkValidDatapoint(datapoint):
                self.df = self.logger.concatResults(self.df, datapoint, [])
                self.logger.saveResults(self.df, self.args)
                continue
            
            if datapoint['input'] is None or datapoint['output'] is None:
                logging.info(f"Datapoint: {datapoint['id']} does not contain input or output")
                continue
            
            logging.info("")
            logging.info(f"Processing datapoint: {datapoint['id']}; {idx}/{data_to_query.shape[0]}")
            strategy_result = self.process_falcon_strategy(datapoint)
            if strategy_result is None:
                logging.info(f"Strategy result is None for datapoint: {datapoint['id']}")
                continue
            queryable_subjects, queryable_objects = strategy_result
            
            # check context for wikipedia
            if (_cntx := datapoint['context']) is not None:
                for cntx in _cntx.split(config.LIST_SEP):
                    if "wikipedia.org" not in cntx:
                        continue
                    
                    get_wikidata_from_wikipedia = self.get_wikidata_from_wikipedia(cntx)
                    if get_wikidata_from_wikipedia is not None and len(get_wikidata_from_wikipedia) > 0:
                        num_wikis_from_context += len(get_wikidata_from_wikipedia)
                        wiki_from_context = f" {config.LIST_SEP} ".join(get_wikidata_from_wikipedia)
                        queryable_objects = f" {config.LIST_SEP} ".join([wiki_from_context, queryable_objects])
                        queryable_subjects = f" {config.LIST_SEP} ".join([wiki_from_context, queryable_subjects])
                        
            
            datapoint['subjects'] = queryable_subjects
            datapoint['objects'] = queryable_objects

            # replace the row in the dataframe
            datapoint = pl.from_dict(datapoint, strict=False)
            data_to_query = data_to_query.update(datapoint, on="id")
            
            # intermediate save: create a dataframe, append newest result, write to disk
            pl.DataFrame({"id": column_order}) \
                .join(
                    pl.concat([data_to_not_query, data_to_query]), 
                    on="id", 
                    how="left") \
                .write_json(f"{self.args.data_dir}/data_falcon_parsed.json")
            
        
        
        # concatenate data and data_non_na
        data = pl.concat([data_to_not_query, data_to_query])
        # sort data by order
        id_df = pl.DataFrame({"id": column_order})
        data = id_df.join(data, on="id", how="left")
        data.write_json(f"deleteme_after.json")
        
        # save
        data.write_json(f"{self.args.data_dir}/data_falcon_fully_parsed.json")
        
        # print stats
        for split in data.group_by('source_dataset'):
            dataset_name = "; ".join(split[0])
            df = split[1]
            subj_is_not_na = df.filter(df['subjects'] != 'N/A')
            subj_is_na = df.filter(df['subjects'] == 'N/A')
            subj_sep_counts = subj_is_not_na.with_columns(subj_is_not_na["subjects"].str.count_matches("<SEP>").alias("sep_count_subj"))['sep_count_subj']

            obj_is_not_na = df.filter(df['objects'] != 'N/A')
            obj_is_na = df.filter(df['objects'] == 'N/A')
            obj_sep_counts = obj_is_not_na.with_columns(obj_is_not_na["objects"].str.count_matches("<SEP>").alias("obj_count_obj"))['obj_count_obj']

            # percentage_subj = (subj_is_not_na.shape[0] / df.shape[0]) * 100
            # percentage_obj = (obj_is_not_na.shape[0] / df.shape[0]) * 100

            subj_data = [("Not N/A", 'N/A'), (subj_is_not_na.shape[0], subj_is_na.shape[0])]
            obj_data = [("Not N/A", 'N/A'), (obj_is_not_na.shape[0], obj_is_na.shape[0])]
            plot_data = {
                "subj": subj_data,
                "subj_core_ents": np.unique(subj_sep_counts, return_counts=True),
                "obj": obj_data,
                "obj_core_ents": np.unique(obj_sep_counts, return_counts=True)
            }
            figs.plot_pie(plot_data, figname=dataset_name, output_dir=self.args.fig_dir + "/entity_parsing")
            
        end_time = time.time()
        logging.info(f"Total time taken: {end_time - start_time}")
        # number of datapoints where subjects is not N/A
        logging.info(f"Total number of datapoints: {data.shape[0]}")
        logging.info(f"Number of datapoints where subjects is not N/A: {data.filter(data['subjects'] != 'N/A').shape[0]}")
        logging.info(f"Number of datapoints where objects is not N/A: {data.filter(data['objects'] != 'N/A').shape[0]}")
        logging.info(f"Number of datapoints where subjects and objects is not N/A: {data.filter((data['subjects'] != 'N/A') & (data['objects'] != 'N/A')).shape[0]}")
        logging.info(f"Found {num_wikis_from_context} wikis from context")
        
        return data
                
                
        
    def checkAnswer(self, sDict, response_entities, datapoint, hop) -> list:
        object_name = response_entities['results']['bindings']
        matches = []
        
        for obj in object_name:
            obj_val = obj['oLabel']['value'].lower().strip()

            for ans_str, a in zip(datapoint['answers_str'], datapoint['answers']):
                ans_str = ans_str.lower().strip()

                # remove starting the
                ans_str = uti.remove_starting_pronouns(ans_str)
                obj_val = uti.remove_starting_pronouns(obj_val)
                    
                if obj_val == ans_str: # hyperparameter, check for similarity threshold?
                    p = obj['p']['value'].split('prop/direct/')[-1]
                    pLabel = self.all_properties[p]['label']
                    o = obj['o']['value'].split('entity/')[-1]
                    oLabel = obj_val

                    pDict = {"p": p, "pLabel": pLabel}
                    oDict = {"o": o, "oLabel": oLabel}
                    triple = self.get_triple(sDict, pDict, oDict)
                    matches.append((f'hop_{hop}', triple))
        
        return matches
    
    
    def sort_wikidata_entities(self, entities):
        entities = [(e.getId().split('entity/')[-1], e.getLabel()) for e in entities]
        entities = list(set(entities))
        return sorted(entities, key=lambda x: int(x[0][1:])) # order them by numerical ID
        
        
    def processSingle(self, input_ent: str, output_ent: str, datapoint) -> dict:
        input_ent_wd = self.api.getKGEntities(input_ent, 50) # hyperparameter: limit retrieve entities by name 1-50
        output_end_wd = self.api.getKGEntities(output_ent, 50)

        # unified entities
        logging.info(f"INPUT Found: {len(input_ent_wd)} entities for input: {input_ent.upper()}")
        logging.info(f"OUTPT Found: {len(output_end_wd)} entities for output: {output_ent.upper()}")

        if len(input_ent_wd) == 0 or len(output_end_wd) == 0:
            return None

        in_ent_sorted = self.sort_wikidata_entities(input_ent_wd)
        out_ent_sorted = self.sort_wikidata_entities(output_end_wd)
        
        queries = [qb.getquery]
        for q in queries:
            res = self._processQuery(q, in_ent_sorted, out_ent_sorted, datapoint, 1, [])
            if res:
                return res
        return None
        # return self._processQuery(entities, datapoint, 1, [], text)
        
                

    def _processQuery(self, query_template, entities, out_entities, datapoint, hop, records, subject):
        """ Breadth first search over entites """
        # Return if we have exhausted the hops or if we have some records
        
        if hop > self.hops or records:
            return records if records else None
        next_level_entities = []  # To collect entities for the next hop

        for e in tqdm(entities, f"Q: {datapoint['input']}; Hop {hop}; Subj: {subject}"):
            s, sLabel = e
            subj_subjLabel = {"s": s, "sLabel": sLabel}
            query = query_template(s)
            response_entities = self.query_engine.send_message(message=query)
            
            # Check if any of the results match the answers
            matches = self.checkAnswer(subj_subjLabel, response_entities, datapoint, hop)
            if matches:
                records.extend(matches)

            # Prepare for the next hop if needed
            for re in response_entities['results']['bindings']:
                oId = re['o']['value'].split('entity/')[-1]
                oLabel = re['oLabel']['value']
                prop_id = re['p']['value'].split('direct/')[-1]
                
                # starts with Q, corresponding property shouldnt be ignored
                # all chars after Q are numeric
                if self.obj_regex(oId) and prop_id not in self.ignore_properties:
                    next_level_entities.append((oId, oLabel))

        # Recursively process the next level of entities
        next_level_entities = list(set(next_level_entities))

        # log hop results
        self.logger.logHopResults(datapoint, entities, next_level_entities, records, subject, hop)
        return self._processQuery(query_template, next_level_entities, datapoint, hop + 1, records, subject)
