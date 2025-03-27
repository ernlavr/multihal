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
                
    
    def choose_best_triples(self, data: pl.DataFrame):
        logging.info("Running choose best trips")
        tmp = data.clone()
        
        for row in tqdm(tmp.iter_rows(named=True), total=len(tmp)):
            # for each row, get the possible tripples
            q = row['input']
            a = row['output']
            trips = row.get('responses').split(config.LIST_SEP)
            trips, trip_labels = self.filter_circular_trips(row)
            mapping = {k:v for k, v in zip(trip_labels, trips)}
            
            # use llm judge to get the most relevant ones
            top_trips = 10
            evaluated_triples = []
            if len(trips) > top_trips:
                attempts = 0
                temperature = self.args.llm_temp
                while top_trips > 1 and attempts < 2:
                    # prompt LLM
                    prompt = self.get_prompt_top_triples(q, a, trip_labels, top_trips)
                    results = llmApi.post_api_request(self.model_name, prompt, temperature)
                    
                    _triples = []
                    for i in results['choices']:
                        content = i['message']['content']
                        for j in content.split("\n"):
                            if "Triple:" in j:
                                j = j.split(":")[-1].strip()
                                if len(j) == 0: continue
                                j = j[1:] if j[0] == '(' else j
                                j = j[:-1] if j[-1] == ')' else j
                                _triples.append(j)
                    
                    if len(_triples) == 0:
                        attempts += 1
                        continue
                    
                    # Map the results back to the originals to avoid hallucinations
                    matched, hallucinated = self.match_labels_to_trip_ids(results, mapping)
                    if len(hallucinated) > 0:
                        logging.info(f"{row['id']}: Hallucinated labels when doing Top-{top_trips} trip extraction: hallc={len(hallucinated)}/{top_trips}; total={len(trip_labels)}")
                    
                    # remove matched out of trip_labels
                    trip_labels = [i for i in trip_labels if i not in matched]
                    evaluated_triples += matched
                    
                    # update counters and pointers
                    top_trips -= len(matched)
                    attempts += 1
                    temperature -= 0.1
                    
                # if we're still missing, randomly sample from the remaining
                if top_trips > 1:
                    if len(trip_labels) > top_trips:
                        sampled = random.sample(trip_labels, top_trips)
                        matched, hallucinated = self.match_labels_to_trip_ids(sampled, mapping)
                    else:
                        matched, hallucinated = self.match_labels_to_trip_ids(trip_labels, mapping)
                    evaluated_triples += matched
                
                
                pairs, _ = self.match_labels_to_trip_ids(evaluated_triples, mapping)
                trip_labels = list(pairs.keys())
                trips = list(pairs.values())
            
            trip_labels = f" {config.LIST_SEP} ".join(trip_labels)
            trips = f" {config.LIST_SEP} ".join(trips)
            
            row['responses'] = trips
            row['trip_labels'] = trip_labels
            
            updated_datapoint = pl.from_dict(row, strict=False)
            data = data.update(updated_datapoint, on="id")
            data.write_json(f"{self.args.data_dir}/llm_judge_results_{self.model_name.replace('/', '-')}_top_trips_int.json")
        
        data.write_json(f"{self.args.data_dir}/llm_judge_results_{self.model_name.replace('/', '-')}_top_trips.json")
        return data
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        # filter out rows which have no triples
        logging.info("Running choose best trips")
        tmp = data.clone()
        output = self.get_results_df()
        
        for row in tqdm(tmp.iter_rows(named=True), total=len(tmp)):
            # for each row, get the possible tripples
            q = row['input']
            a = row['output']
            opt_a = row.get('optional_output')
            if opt_a is not None:
                opt_a = opt_a.split(config.LIST_SEP)
                a = [a] + opt_a
            # trips = row.get('responses').split(config.LIST_SEP)
            trip_labels = row.get('trip_labels').split(config.LIST_SEP)
            trip_codes = row.get('responses').split(config.LIST_SEP)
            
            trips = zip(trip_labels, trip_codes)
            logging.info(f"Processing row {row['id']} with triples (n={len(trip_codes)})")
            
            for label, code in trips:
                if len(label) == 0: continue
                # for each triple, decode the identifiers to labels    
                # labels = self.kg_manager.decode_statement_labels(trip.split())
                label = label.strip()
                code = code.strip()
                # run inference on the triples' relevance to the question with respect to the expected answer
                prompt = self.get_prompt_triple_relevance(q, a, label)
                temp = self.args.llm_temp
                request_response = llmApi.post_api_request(self.model_name, prompt, temp, max_tokens=256)
                
                score = -1
                for i in request_response['choices']:
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
                        
                
                entry = jbc.JudgeEvalResult(
                    id=row['id'],
                    source_dataset=row['source_dataset'],
                    task=row['task'],
                    domain=row['domain'],
                    input=q,
                    output=a,
                    responses=code,
                    trip_labels=label,
                    judged_by=self.model_name,
                    judged_label="N/A",
                    judged_score=score
                )
                
                # save the output to a dict
                output = self.add_result(entry, output)
            output.write_json(f"{self.args.data_dir}/llm_judge_trip_eval_{self.model_name.replace('/', '-')}_int.json")
        
        output.write_json(f"{self.args.data_dir}/llm_judge_trip_eval_{self.model_name.replace('/', '-')}_final.json")
        return output
    
    def evaluate(self, data):
        pass