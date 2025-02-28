import spacy

import src.kgs.querybuilder as qb
import src.network.api.local_api as api
import src.network.udp_manager as br
import src.utils.helpers as uti
import src.utils.logger as log
import src.utils.config as config
import src.analysis.figures as figs

import time
import requests
import json
import polars as pl
import numpy as np
import re
import logging
from itertools import chain

from tqdm import tqdm

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
        
        # Logging
        self.ignore_properties = uti.fill_ignore_properties("res/wd_properties_to_ignore/ids_to_remove.json")
        self.all_properties = uti.fill_all_properties('res/wd_properties_to_ignore/properties.json')

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
    
    def decode_statement_labels(self, statement: list):
        labels = []
        for entity in statement:
            if entity.startswith('Q') and len(entity) < 14:
                query = qb.get_label_of_entity(entity)
                response = self.bridge.send_message(message=query)
                labels.append(response['results']['bindings'][0]['label']['value'])
            elif entity.startswith('P'):
                property = self.all_properties.get(entity, None)
                if property is not None:
                    property = property['label']
                labels.append(property)
        
        if len(statement) != len(labels):
            logging.error(f"Length of statement: {len(statement)} does not match length of labels: {len(labels)}; statement: {statement}; labels: {labels}")
        return labels
            
    
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

    def get_pos_rules(self, sequences, include_children=False):
        """ Rules to extract the entity 
            1. No stop-words
            2. Compare token for POS and DEP tags
                2.1. Optionally check token parse tree for child nodes
        """
        output = ""
        allowed_token_dep = ['nsubj', 'dobj', 'obj', 'iobj', 'pobj']
        allowed_child_dep = ['amod', 'nmod', 'appos', 'compound']
        allowed_token_pos = ['NOUN', 'PROPN']
        output = []

        for sequence in sequences:
            for token in sequence:
                # skip stop words
                if token.is_stop:
                    continue
                # skip non-nouns
                if token.dep_ not in allowed_token_dep and token.pos_ not in allowed_token_pos:
                    continue
                
                # We're definitely adding token
                entry = token

                # add dependency modifier, i.e. adjectives, maybe adding children
                if include_children:
                    children = []
                    for child in token.subtree:
                        if child.dep_ not in allowed_child_dep or child is entry: 
                            continue
                        children.append(child)
                    children.append(entry)
                    entry = tuple(children)
                
                # continue if entry is already in output
                if self.tuple_exists_by_text(output, entry): continue
                output.append(entry)
        return output
        
    def get_falcon_query(self, query_text, timeout=5):
        url = "https://labs.tib.eu/falcon/falcon2/api?mode=long&k=5"
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
        entities_wikidata = response.get('entities_wikidata') or []
        combined_entities = entities_dbpedia + entities_wikidata
        if len(combined_entities) == 0:
            logging.info(f"No entities found in either Wikidata or Dbpedia for query: {query_text}")
            return ["<NO_ENTITIES_FOUND>"]

        for i in combined_entities:
            output.append(i['URI'])
        
        return output
    
    def process_queryable_entities(self, entity, entity_type, timeout=60):
        if 'N/A' in entity:
            query_result = self.get_falcon_query(entity_type)
            if entity is None:
                logging.info(f"Error when querying Falcon API for {entity_type}, returned None.")
                return None
        
        elif f"<TIMEOUT>{entity_type}</TIMEOUT>" in entity:
            logging.info(f"Timeout when querying Falcon API for {entity_type}.")
            query_result = self.get_falcon_query(entity_type, timeout=timeout)
            if query_result is None:
                logging.info(f"Error when querying Falcon API for {entity_type}, returned None.")
                return None
        else:
            return entity.split(config.LIST_SEP)
            
        # Handle if a timeout happened in a list (e.g. objects)
        if config.LIST_SEP in entity:
            entity = entity.split(config.LIST_SEP).extend(query_result)
        else:
            entity = query_result
        
        return list(set(entity))  # Remove duplicates
    
    def process_falcon_strategy(self, datapoint):
        # Process queryable subjects
        queryable_subjects = self.process_queryable_entities(datapoint['subjects'], datapoint['input'])
        if queryable_subjects is None:
            return None

        # Process queryable objects
        outputs = [datapoint['output']]
        if datapoint['optional_output'] is not None:
            outputs.extend(datapoint['optional_output'].split(config.LIST_SEP))
        
        outputs = list(set(outputs))  # Remove duplicates
        if len(outputs) > 3:
            outputs = outputs[:3]
            logging.info(f"Truncated outputs to 3 for datapoint: {datapoint['id']}")
        
        queryable_objects = []
        for output in outputs:
            qs = self.get_falcon_query(output)
            if qs is None:
                logging.info(f"Error when querying Falcon API for output ID {datapoint['id']}: {output}")
                continue
            queryable_objects.extend(qs)
        
        queryable_objects = uti.flatten_if_2d(queryable_objects)
        queryable_objects = list(set(queryable_objects))  # Remove duplicates

        return (
            f" {config.LIST_SEP} ".join(queryable_subjects),
            f" {config.LIST_SEP} ".join(queryable_objects)
        )
    
    def process_spacy_strategy(self, datapoint):
        # make it a list, because we may have multiple sequences in the generic function
        utterance = [self.spacy_model(datapoint['input'])]

        if (t := datapoint['optional_output']) is not None: 
            output_ents = t.split(config.LIST_SEP)
            if datapoint['output'] not in output_ents:
                output_ents.extend([datapoint['output']])
            output_ents = [self.spacy_model(i) for i in output_ents]
        else: 
            output_ents = [self.spacy_model(datapoint['output'])]
            
        
        utterance_tok = self.get_pos_rules(utterance, include_children=True)
        output_ents_tok = self.get_pos_rules(output_ents, include_children=True)

        logging.info(f"Datapoint: {datapoint['id']}")
        logging.info(f"Input: {datapoint['input']}")
        logging.info(f"Input Tags: {[(j.text, j.dep_) for i in utterance for j in i]}")
        logging.info("")
        if datapoint['optional_output'] is not None:
            logging.info(f"Output: {datapoint['output'] + config.LIST_SEP + datapoint['optional_output']}")
        else:
            logging.info(f"Output: {datapoint['output']}")
        logging.info(f"Output Tags: {[(j.text,  j.dep_) for i in output_ents for j in i]}")
        logging.info("")
        logging.info(f"Input Entities: {utterance_tok}")
        logging.info(f"Output Entities: {output_ents_tok}")
        logging.info("")
        logging.info("")

        # flatten the spacy parsing with <SEP> inbetween
        ut = 'N/A'
        oet = 'N/A'
        if len(utterance_tok) is not 0:
            ut = f"{config.LIST_SEP}".join([" ".join([ut.text for ut in uts]) for uts in utterance_tok])
        if len(output_ents_tok) is not 0:
            oet = f"{config.LIST_SEP}".join([" ".join([oet.text for oet in oets]) for oets in output_ents_tok])

        return ut, oet
    
    def _get_kg_so_pairs(self, datapoint: dict, kg_name="wikidata"):
        """ Returns a list of tuples that has all pair-wise combinations between subject-object entities """
        # get the entities
        subjects = datapoint.get("subjects")
        objects = datapoint.get("objects")
        if subjects is None or objects is None:
            return []
        
        # Subjects/objects may be seperated by <SEP>
        subjects = subjects.split(config.LIST_SEP)
        objects = objects.split(config.LIST_SEP)

        # Finally get the pairs
        pairs = []
        for s in subjects:
            if kg_name not in s:
                continue

            for o in objects:
                if kg_name not in o:
                    continue
                
                # extract only the QXXXX entity identifiers
                if 'wikidata' in kg_name:
                    s = s.split('/')[-1].strip()
                    o = o.split('/')[-1].strip()

                if 'dbpedia' in kg_name: # TODO cross-check this
                    raise NotImplementedError("DBPedia not implemented yet")

                pairs.append((s, o))
        return pairs

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
                filtered = [item for item in i.keys() if "Label" not in item]
                filtered.sort(key=lambda x: (int(x[1:]), x[0] == 'o'))
                path = ""
                for j in filtered:
                    if 'Label' in j:
                        continue
                    else:
                        path += f"{i[j]['value']} ".split('/')[-1]
                paths += f"{subj} {path} {obj} {config.LIST_SEP}"

        return paths

    def query_kg(self, data: pl.DataFrame, network_bridge: br.NetworkBridge, max_hops=3):
        progress_bar = tqdm(data.iter_rows(named=True), total=data.shape[0])
        # datapoint add column responses
        data = data.with_columns(
            responses=pl.lit('N/A')
        )
        total_queries = 0

        for datapoint in progress_bar:
            logging.info(f"Processing Datapoint: {datapoint['id']}")
            progress_bar.set_description(f"Querying Wikidata")

            # pair-wise path search between subject-object entities
            pairs = self._get_kg_so_pairs(datapoint)
            if not pairs:
                continue

            responses = []
            for subj, obj in pairs:
                total_queries += 1
                curr_hop = 1
                while curr_hop <= max_hops:
                    # plug in the pair in the query
                    query = qb.get_query_so_hops(subj, obj, curr_hop)
                    
                    # run the query and save the response
                    response = network_bridge.send_message(message=query)
                    if response:
                        parsed = self.parse_response(subj, obj, response)
                        if len(parsed) > 0:
                            responses.append(parsed)
                    # increase hop
                    curr_hop += 1
            
            responses = " ".join(responses).strip()
            # remove last <SEP>
            if responses.endswith(config.LIST_SEP):
                responses = responses[:-len(config.LIST_SEP)]

            # add responses to the dataframe
            datapoint['responses'] = responses
            _datapoint = pl.from_dict(datapoint, strict=False)
            data = data.update(_datapoint, on="id")
            data.write_json(f"{self.args.data_dir}/data_subj_obj_parsed_queried_wd.json")
        logging.info(f"Total number of queries: {total_queries}")
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
        
        for datapoint in tqdm(data_to_query.iter_rows(named=True), "Processing Dataset", total=data_to_query.shape[0]):
            if not self.checkValidDatapoint(datapoint):
                self.df = self.logger.concatResults(self.df, datapoint, [])
                self.logger.saveResults(self.df, self.args)
                continue
            
            if datapoint['input'] is None or datapoint['output'] is None:
                logging.info(f"Datapoint: {datapoint['id']} does not contain input or output")
                continue
            
            logging.info("")
            logging.info(f"Processing datapoint: {datapoint['id']}")
            strategy_result = self.process_falcon_strategy(datapoint)
            if strategy_result is None:
                logging.info(f"Strategy result is None for datapoint: {datapoint['id']}")
                continue
           
            queryable_subjects, queryable_objects = strategy_result
            datapoint['subjects'] = queryable_subjects
            datapoint['objects'] = queryable_objects

            # replace the row in the dataframe
            datapoint = pl.from_dict(datapoint, strict=False)
            data_to_query = data_to_query.update(datapoint, on="id")
            continue
        
        
        # concatenate data and data_non_na
        data = pl.concat([data_to_not_query, data_to_query])
        # sort data by order
        id_df = pl.DataFrame({"id": column_order})
        data = id_df.join(data, on="id", how="left")
        data.write_json(f"deleteme_after.json")
        
        # save
        data.write_json(f"{self.args.data_dir}/data_falcon_parsed.json")
        
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
