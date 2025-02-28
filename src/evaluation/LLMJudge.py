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


class LLMJudge():
    def __init__(self, hf_model, args):
        self.args = args
        self.kg_manager = kgm.KGManager(None, args)
        if hf_model is not None:
            utils.print_cuda_stats()
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model, self.device = self.get_model(hf_model)
            self.run_inference("Best ways of handling business is by")
    
    def get_model(self, model_name, distributed=False) -> tuple[AutoModelForCausalLM, torch.device]:
        # Initialize distributed 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (rank := os.environ.get("RANK", None)) is not None:
            rank = int(rank)
            device = torch.device(f"cuda:{rank}")
            torch.distributed.init_process_group("nccl", device_id=device)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            tp_plan="auto" if rank is not None else None,
        )
        return model, device
    
    def get_instructions(self):
        """ Returns instruction prompt for the LLM """
        instructions = "Continue the following sentence: \n"
        pass
    
    def get_prompt_triple_relevance(self, question, answer, triple):
        """ Returns the prompt for the LLM to evaluate the relevance of the triple to the question and answer """
        instructions = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You need to evaluate whether a given Wikidata triple is informative to the given question with respect to the expected answer.
            Output in YAML format one of three options, "Yes", "No", "Unsure" and indicate your confidence level.
            
            Here is an expected format of the input:
            Question: What is the capital of France?
            Answer: Paris
            Triple: (France, capital, Paris)
            
            Here is an example of the output
            Relevance: Yes
            Confidence: 0.9
            <|eot_id|>
            
            <|start_header_id|>user<|end_header_id|>
            Question: {question}
            Answer: {answer}
            Triple: ({", ".join(triple)})
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
        """
        
        return instructions
    
    def run_inference(self, text):
        """ Expects formatted text """
        # Run inference
        inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        outputs = self.model.generate(
            inputs, 
            max_new_tokens=256,
            do_sample=False,     # No randomness
            num_beams=5,         # Use beam search for better coherence
            temperature=0.3,     # Low temperature for precise answers
            repetition_penalty=1.2
        )

        # Decode
        output_text = self.tokenizer.decode(outputs[0])
        return output_text
    
    def extract_relevance_confidence(self, text):
        relevance_match = re.search(r"Relevance:\s*(\w+)", text)
        confidence_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", text)
        
        relevance = relevance_match.group(1) if relevance_match else None
        confidence = float(confidence_match.group(1)) if confidence_match else None
        
        return relevance, confidence
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        # filter out rows which have no triples
        _data = data.filter(~data['responses'].is_in(['N/A', ""]))
        output = {}
        relevances = []
        for row in tqdm(_data.iter_rows(named=True)):
            # for each row, get the possible tripples
            q = row['input']
            a = row['output']
            opt_a = row.get('optional_output')
            if opt_a is not None:
                opt_a = opt_a.split(config.LIST_SEP)
                a = [a] + opt_a
            trips = row.get('responses').split(config.LIST_SEP)
            
            output[row['id']] = []
            for trip in trips:
                if len(trip) == 0: continue
                # for each triple, decode the identifiers to labels    
                labels = self.kg_manager.decode_statement_labels(trip.split())
                # run inference on the triples' relevance to the question with respect to the expected answer
                prompt = self.get_prompt_triple_relevance(q, a, labels)
                result = self.run_inference(prompt)
                result = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                
                relevance, confidence = self.extract_relevance_confidence(result)
                
                # save the output to a dict
                output[row['id']].append({'relevance': relevance, 'confidence': confidence})
                relevances.append(relevance)
                
        return output, relevances
    
    def evaluate(self, data):
        pass