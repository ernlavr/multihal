from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import src.utils.helpers as utils
import os
import polars as pl
from tqdm import tqdm
import src.utils.config as config
import src.kgs.kg_manager as kgm
import src.network.LlmApi as llmApi
import re
import transformers
import src.evaluation.JudgeBaseClass as jbc
import requests
import json
import numpy as np
import random
import time
import copy

class API_Judge(jbc.JudgeBaseClass):
    def __init__(self, model_name, args):
        self.args = args
        self.model_name = model_name
            
    def extract_relevance_confidence(self, text):
        trip = re.search(r"Triple:\s*(\w+)", text)
        score = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", text)
        
        trip = trip.group(1) if trip else None
        score = float(score.group(1)) if score else None
        
        return trip, score
    
    def match_labels_to_trip_ids(self, trip_labels, mapping):
        output = {}
        mismatched = []
        
        # lowercase mapping keys
        mapping = {k.lower(): v for k, v in mapping.items()}
        
        for label in trip_labels:
            tmp_label = label.lower()
            if tmp_label in mapping:
                output[label] = mapping[tmp_label]
            else:
                mismatched.append(label)
        return output, mismatched
    
    def parse_api_results(self, results):
        paths = []
        if results is None:
            return paths
        
        for i in results['choices']:
            content = i['message']['content']
            for j in content.split("\n"):
                if "Path:" in j:
                    j = j.split(":")[-1].strip()
                    if len(j) == 0: continue
                    j = j[1:] if j[0] == '(' else j
                    j = j[:-1] if j[-1] == ')' else j
                    paths.append(j)
        return paths
    
    def get_random_trip_labels(self, row: dict, intersecting_trips, total_samples):
        trip_labels: list = row.get('trip_labels').split(config.LIST_SEP)
        trip_labels = [i.replace("_", " ") for i in trip_labels]
        trip_codes = row.get('responses_formatted').split(config.LIST_SEP)
        mapping = {k:v for k, v in zip(trip_labels, trip_codes)}
        
        for it in intersecting_trips:
            trip_labels.remove(it)
            
        num_samples = total_samples - len(intersecting_trips)
        if num_samples <= 0:
            return intersecting_trips, []
        
        # randomly sample from the remaining trip_labels
        sampled = []
        if len(trip_labels) > num_samples:
            sampled = random.sample(trip_labels, num_samples)
        else:
            sampled = trip_labels
            
        trip_codes = [mapping[i] for i in sampled]

        return sampled, trip_codes
            
            
            
        
    
    def _get_best_trips(self, row, top_trips=10, num_shuffles=1):
        q = row['input']
        a = row['output']

        output = []
        for shuffle_idx in range(num_shuffles + 1):
            trips = row.get('responses_formatted').split(config.LIST_SEP)
            trip_labels = row.get('trip_labels').split(config.LIST_SEP)
            # replace "_" with " "
            trip_labels = [i.replace("_", " ") for i in trip_labels]
            
            
            mapping = {k:v for k, v in zip(trip_labels, trips)}
            
            assert len(trips) == len(trip_labels), f"Trip labels and trip codes do not match: {len(trips)} != {len(trip_labels)}"
            
            top_trips = 10
            attempts = 0
            temperature = self.args.llm_temp
            evaluated_triples = []
            while top_trips >= 1 and attempts < 2:
                # shuffle the trip labels and codes for LLM-Judge methodology        
                shuffled_labels = trip_labels.copy()
                random.shuffle(shuffled_labels)
                
                prompt = self.get_prompt_top_triples(q, a, shuffled_labels, top_trips)
                results = llmApi.post_api_request(self.model_name, prompt, temperature)
                results = self.parse_api_results(results)
                
                # if we fail to get some results
                if results is None or len(results) == 0:
                    attempts += 1
                    temperature -= 0.05
                    continue
                
                # Map the results back to the originals to avoid hallucinations
                matched, hallucinated = self.match_labels_to_trip_ids(results, mapping)
                if len(hallucinated) > 0:
                    logging.info(f"{row['id']}: Hallucinated labels when doing Top-{top_trips} trip extraction: hallc={len(hallucinated)}/{top_trips}; total={len(trip_labels)}")
                    logging.info("\n".join(hallucinated))
                
                # remove matched out of trip_labels
                trip_labels = [i for i in trip_labels if i not in matched]
                evaluated_triples += matched
                
                # update counters and pointers
                top_trips -= len(matched)
                attempts += 1
                temperature -= 0.05
                
            # if we're still missing, randomly sample from the remaining
            if top_trips >= 1:
                if len(trip_labels) > top_trips:
                    sampled = random.sample(trip_labels, top_trips)
                    matched, hallucinated = self.match_labels_to_trip_ids(sampled, mapping)
                    logging.info(f"{row['id']}: Randomly sampled {top_trips} trip labels from {len(trip_labels)} remaining")
                else:
                    matched, hallucinated = self.match_labels_to_trip_ids(trip_labels, mapping)
                evaluated_triples += matched
            
            
            pairs, _ = self.match_labels_to_trip_ids(evaluated_triples, mapping)
            selected_trip_labels = list(pairs.keys())
            selected_trip_codes = list(pairs.values())
            
            output.append({
                "trip_labels": selected_trip_labels,
                "responses_formatted": selected_trip_codes,
            })
        
        return output
    
    def choose_best_triples(self, data: pl.DataFrame):
        logging.info("Running choose best trips")
        tmp = data.clone()
        intersecting_dp_count = []
        save_path = f"{self.args.data_dir}/llm_judge_trip_sel_{self.model_name.replace('/', '-')}.json"
        random_fills = 0
        total_paths = 0
        
        # filter for more than 10 <SEP>
        tmp = tmp.filter(pl.col('responses_formatted').str.count_matches(config.LIST_SEP) > 10)
        
        for row in tqdm(tmp.iter_rows(named=True), total=len(tmp)):
            top_trips = 10
            path_counts = len(row['responses_formatted'].split(config.LIST_SEP))
            
            if path_counts <= top_trips:
                total_paths += path_counts
                continue
            total_paths += top_trips
            
            best_trips = self._get_best_trips(row, num_shuffles=1)
            intersection = set(best_trips[0]['trip_labels']).intersection(set(best_trips[1]['trip_labels']))
            intersecting_dp_count.append(len(intersection))
            
            # if we dont have enough in our intersection, fill the gap randomly
            trips = []
            trip_labels = []
            
            if len(intersection) < top_trips:
                random_fills += top_trips - len(intersection)
                trip_labels, trips = self.get_random_trip_labels(row, intersection, top_trips)    
            
            for i in intersection:
                idx = best_trips[0]['trip_labels'].index(i)
                trips.append(best_trips[0]['responses_formatted'][idx])
                trip_labels.append(best_trips[0]['trip_labels'][idx])
                
            
            trip_labels = f"{config.LIST_SEP}".join(trip_labels)
            trips = f"{config.LIST_SEP}".join(trips)
            
            row['responses_formatted'] = trips
            row['trip_labels'] = trip_labels
            
            updated_datapoint = pl.from_dict(row, strict=False)
            data = data.update(updated_datapoint, on="id")
            data.write_json(save_path.replace(".json", "_step.json"))
        
        data.write_json(save_path)
        logging.info(f"Total paths: {total_paths}; Randomly filled: {random_fills}")
        logging.info(f"Intersecting datapoints (dp={len(intersecting_dp_count)} / {len(tmp)}): mean={np.mean(intersecting_dp_count)}; std={np.std(intersecting_dp_count)}")
        logging.info(f"Saved to {save_path}")
        return data
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        # filter out rows which have no triples
        logging.info("Running choose best trips")
        output = data.clone()
        if 'judged_by' not in output.columns and 'judged_score' not in output.columns:
            output = output.with_columns([
                pl.lit("<NOT_JUDGED>").alias("judged_by"),
                pl.lit(None, dtype=pl.Int32).alias("judged_score")
            ])
            
        # Get unprocessed datapoints
        output = output.filter(pl.col('judged_by') ==  "<NOT_JUDGED>")
        
        save_path = f"{self.args.data_dir}/llm_judge_trip_rate_{self.model_name.replace('/', '-')}.json"
        
        for row in tqdm(output.iter_rows(named=True), total=len(output)):
            # for each row, get the possible tripples
            q = row['input']
            a = row['output']
            opt_a = row.get('optional_output')
            if opt_a is not None:
                opt_a = opt_a.split(config.LIST_SEP)
                a = [a] + opt_a
            # trips = row.get('responses').split(config.LIST_SEP)
            trip_labels = row.get('trip_labels').split(config.LIST_SEP)
            trip_codes = row.get('responses_formatted').split(config.LIST_SEP)
            
            if trip_labels == 'N/A' or trip_labels == '':
                continue
            
            trips = zip(trip_labels, trip_codes)
            logging.info(f"Processing row {row['id']} with triples (n={len(trip_codes)})")
            
            for idx, (label, code) in enumerate(trips):
                if len(label) == 0: continue
                # for each triple, decode the identifiers to labels    
                # labels = self.kg_manager.decode_statement_labels(trip.split())
                label = label.strip()
                code = code.strip()
                # run inference on the triples' relevance to the question with respect to the expected answer
                prompt = self.get_prompt_triple_relevance(q, a, label)
                temp = self.args.llm_temp
                request_response = llmApi.post_api_request(self.model_name, prompt, temp, max_tokens=256)
                if request_response is None:
                    logging.error(f"{row['id']}: API request returned None")
                    continue
                
                score = -1
                for i in request_response['choices']:
                    if len(request_response['choices']) > 1:
                        logging.error(f"{row['id']}: More than 1 response from LLM: {len(request_response['choices'])}")
                        continue
                    
                    content = i['message']['content'].lower()
                    if "score:" not in content.lower():
                        logging.info(f"{row['id']}: Failed to parse score: {content}")
                        continue
                    
                    score = content.split("score:")[-1].strip()
                    if len(score) == 0: continue
                    # try to parse the score
                    try:
                        score = int(score)
                    except:
                        logging.error(f"{row['id']}: Failed to parse score: {content}")
                        score = -1
                
                entry = copy.deepcopy(row)
                entry['id'] = f"{entry['id']}_{idx}"
                entry['judged_by'] = self.model_name
                entry['judged_score'] = score
                entry['responses_formatted'] = code
                entry['trip_labels'] = label
                
                # save the output to a dict
                output = self.add_result(entry, output)
            output.write_json(save_path.replace(".json", "_step.json"))
        
        output.write_json(save_path)
        return output
    
    def evaluate(self, data):
        pass