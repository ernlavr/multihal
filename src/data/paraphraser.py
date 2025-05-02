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
from collections import defaultdict

class Paraphraser():
    def __init__(self, args):
        self.args = args
        self.filter_until_alpha_pattern = re.compile(r'^[^a-zA-Z]+')
    
    def strip_until_alpha(self, s):
        return self.filter_until_alpha_pattern.sub('', s)
    
    def parse_results(self, llm_api_response):
        """ Sends a request to the LLM API and returns the response """
        content = None
        try:
            content = llm_api_response['choices'][0]['message']['content']
        except:
            logging.error(f"Error parsing paraphrased LLM API response {llm_api_response}")
            return None
        
        content = content.split('\n')
        if len(content) == 0:
            logging.error(f"Error splitting LLM paraphrased content {content}")
            return None
        
        # strip all non alpha characters from the start        
        return [self.strip_until_alpha(x.strip()) for x in content if x.strip()]
    
    def get_dp_without_paraphrases(self, df: pl.DataFrame, df_pp: pl.DataFrame) -> pl.DataFrame:
        # get the intersection between df and df_pp based on IDs
        return df.join(df_pp, on='id', how='anti')
    
    def cleanup_paraphrase_df(self, df_pp: pl.DataFrame) -> pl.DataFrame:
        df_pp = df_pp.select(['id', 'input']).rename({'id': 'paraphrase_of'})

        # Add an index per group using `cumcount` to create unique IDs
        df_pp = df_pp.with_columns([
            pl.col('paraphrase_of'),
            pl.col('input'),
            pl.col('paraphrase_of').cum_count().over('paraphrase_of').cast(pl.Utf8).alias('group_index')
        ])

        df_pp = df_pp.with_columns([
            (pl.col('paraphrase_of') + "_" + pl.col('group_index')).alias('id')
        ]).select(['id', 'paraphrase_of', 'input'])

        return df_pp
        

    def generate_paraphrasings(self, df, df_pp):
        # remove all columns from df_pp except id and input
        unprocessed_datapoints = self.get_dp_without_paraphrases(df, df_pp)
        df_pp = self.cleanup_paraphrase_df(df_pp)
        df_pp = df_pp.with_columns(
            rephrased_by=pl.lit('default_from_source')
        )
        rephrasing_model = self.args.llm_judge_model.replace('/', '-')
        save_dir = os.path.join(self.args.data_dir, f'{rephrasing_model}_paraphrased.json')       
        
                
        
        for i, row in enumerate(tqdm(unprocessed_datapoints.iter_rows(named=True), desc="Matching paraphrasings")):
            # Get the main datapoint
            question = row['input']
            prompt = self.get_prompt_paraphrasing(question)
            response = llmApi.post_api_request(self.args.llm_judge_model, prompt, self.args.llm_temp, max_tokens=2048)
            results = self.parse_results(response)
            
            
            entry = defaultdict(list)
            for idx, paraphrase in enumerate(results):
                
                entry['id'].append(f"{row['id']}_{idx}")
                entry['paraphrase_of'].append(row['id'])
                entry['input'].append(paraphrase)
                entry['rephrased_by'].append(rephrasing_model)
                
                
                
            df_pp = df_pp.vstack(pl.DataFrame(entry))
            df_pp.write_json(os.path.join(self.args.data_dir, f'{rephrasing_model}_paraphrased.json'))
        
        df.write_json(os.path.join(self.args.data_dir, f'multihal_paraphrase_sources.json'))
        df_pp.write_json(os.path.join(self.args.data_dir, f'{rephrasing_model}_paraphrased_final.json'))
        logging.info(f"Finished paraphrasing {len(unprocessed_datapoints)} datapoints")
        logging.info(f"Final paraphrased dataset size: {len(df_pp)}")
        logging.info(f"Paraphrased dataset saved to {save_dir}")
        return df, df_pp
        
                
        
        
    def get_prompt_paraphrasing(self, question):
        """ Returns the prompt for the LLM to evaluate the relevance of the triple to the question and answer """
        # weird indenting is for formatting
        messages = [
            {"role": "user", 
             "content": f""" You need to generate 15 paraphrased versions of the given question. They need to be semantically the same, just with different wordings.
             
Here is an example input and output:
Question: What was the population count of Australia's 0-4 age group in 2011?;
Paraphrases:
- How many individuals aged 0-4 were there in Australia in 2011?
- What was Australia's population figure for the 0-4 age bracket in 2011?
- In 2011, what was the number of people aged 0-4 residing in Australia?
- What was the demographic size of Australia's 0-4 age category in 2011?
- How many residents aged 0-4 were recorded in Australia in 2011?
- What was the population tally for Australia's 0-4 age group in 2011?
- How many individuals aged 0-4 were living in Australia in 2011?
- What was the count of Australia's 0-4 age range population in 2011?
- What was the population size of Australia's 0-4 age group in 2011?
- What was the population census result for Australia's 0-4 age cohort in 2011?
- How many people in the age group 0-4 were accounted for in Australia's 2011 census?
- What was the population total for Australia's 0-4 age demographic in 2011?
- In 2011, what was the population of Australia of the age group 0-4 years?
- Population of Australia in 2011 between 0-4 age group?
- What was the number of individuals aged 0-4 in Australia in 2011?

Now perform the paraphrasing of the following question:
Question: {question};
Paraphrases:"""}]
        return messages