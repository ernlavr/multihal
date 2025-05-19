from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import src.utils.helpers as utils
import os
import polars as pl
from tqdm import tqdm
import src.utils.config as config
import src.kgs.kg_manager as kgm
import re
import transformers
import src.evaluation.JudgeBaseClass as jbc
import random

class LLMJudge(jbc.JudgeBaseClass):
    def __init__(self, model_name, args):
        self.args = args
        self.kg_manager = kgm.KGManager(None, args)
        self.model_name = model_name
        if model_name is not None:
            utils.print_cuda_stats()
            self.model, self.tokenizer = self.get_pipeline(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_pipeline(self, model_name) -> tuple[AutoModelForCausalLM, torch.device]:
        # Initialize distributed 
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
                
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            max_new_tokens=256,
            device_map="auto",
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            return_full_text=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        return pipeline, tokenizer, device
    
    def extract_relevance_confidence(self, text):
        trip = re.search(r"Triple:\s*(\w+)", text)
        score = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", text)
        
        trip = trip.group(1) if trip else None
        score = float(score.group(1)) if score else None
        
        return trip, score
    
    
    def choose_best_triples(self, data: pl.DataFrame):
        """ LLM as a judge for selection """
        logging.info("Running choose best trips")
        unprocessed_ents, processed_ents = self.split_unprocessed_ents(data)
        output = self.get_results_df()
        output_path = f"{self.args.data_dir}/llm_judge_results_{self.model_name.replace('/', '-')}"
        
        for row in tqdm(unprocessed_ents.iter_rows(named=True), total=len(unprocessed_ents)):
            # for each row, get the possible tripples
            q = row['input']
            a = row['output']
            trips = row.get('responses').split(config.LIST_SEP)
            trip_labels = row.get('trip_labels').split(config.LIST_SEP)
            
            combined = list(zip(trips, trip_labels))
            random.shuffle(combined)
            
            shuffled_trips, shuffled_labels = zip(*combined)
            shuffled_trips, shuffled_labels = list(shuffled_trips), list(shuffled_labels)
            
            # run inference on the triples' relevance to the question with respect to the expected answer
            prompt = self.get_prompt_top_triples(q, a, trip_labels)
            result = self.pipeline(prompt)
            
            prompt_shuffled = self.get_prompt_top_triples(q, a, shuffled_labels)
            result_shuffled = self.pipeline(prompt_shuffled)
            
            try:
                trip, score = self.extract_relevance_confidence(result)
            except:
                logging.info(f"Model did not output parsable response: {result}")
                trip, score = "<FAILED>", -1.0
            
            entry = jbc.JudgeEvalResult(
                id=row['id'],
                source_dataset=row['source_dataset'],
                task=row['task'],
                domain=row['domain'],
                input=q,
                output=a,
                responses=trips,
                trip_labels=trip_labels,
                judged_by=self.model_name,
                judged_label=trip,
                judged_score=score
            )
            output = self.add_result(entry, output)
            output.write_json(f"{output_path}_top_trips_int.json")
        
        logging.info(f"Processed entities: {len(unprocessed_ents)}/{len(data)}")
        save_path = f"{self.args.data_dir}/llm_judge_trip_selection{self.model_name.replace('/', '-')}.json"
        output.write_json(save_path)
        return output
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        """ LLM as a judge for rating """
        # filter out rows which have no triples
        unprocessed_ents = self.get_unprocessed_datapoints(data)
        output = self.get_results_df()
        relevances = []
        
        for row in tqdm(unprocessed_ents.iter_rows(named=True)):
            # for each row, get the possible tripples
            q = row['input']
            a = row['output']
            opt_a = row.get('optional_output')
            if opt_a is not None:
                opt_a = opt_a.split(config.LIST_SEP)
                a = [a] + opt_a
            # trips = row.get('responses').split(config.LIST_SEP)
            trips = row.get('trip_labels').split(config.LIST_SEP)
            logging.info(f"Processing row {row['id']} with triples (n={len(trips)})")
            
            for trip in trips:
                if len(trip) == 0: continue
                # for each triple, decode the identifiers to labels    
                # labels = self.kg_manager.decode_statement_labels(trip.split())
                labels = trip
                # run inference on the triples' relevance to the question with respect to the expected answer
                prompt = self.get_prompt_triple_relevance(q, a, labels)
                result = self.run_inference(prompt)
                result = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                
                relevance, confidence = self.extract_relevance_confidence(result)
                entry = pl.DataFrame({'id': row['id'], 
                                      'source_dataset': row['source_dataset'], 
                                      'domain': row['domain'], 
                                      'input': q, 
                                      'output': f"{config.LIST_SEP}".join(a), 
                                      'trip': trip, 
                                      'trip_labels': f" ".join(labels), 
                                      'relevance': relevance, 
                                      'confidence': confidence})
                
                # save the output to a dict
                output = pl.concat([output, entry])
                relevances.append(relevance)
            output.write_csv(f"{self.args.data_dir}/llm_judge_results_intermediate.csv")
        
        output.write_csv(f"{self.args.data_dir}/llm_judge_results.csv")
        return output, relevances
    
    def evaluate(self, data):
        pass