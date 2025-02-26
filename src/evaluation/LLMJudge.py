import transformers
import torch
import logging
import src.utils.helpers as utils
import os
import polars as pl
from tqdm import tqdm
import src.utils.config as config

class LLMJudge():
    def __init__(self, hf_model, args):
        self.args = args
        
        if hf_model is not None:
            utils.print_cuda_stats()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
            self.model, self.device = self.get_distributed_model(hf_model)
            self.run_inference("Best ways of handling business is by")
    
    def get_distributed_model(self, model_name):
        # Initialize distributed
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{rank}")
        torch.distributed.init_process_group("nccl", device_id=device)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            tp_plan="auto",
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
            Triple: {triple}
            <|eot_id|>
            
            <|start_header_id|>assistant<|end_header_id|>
        """
        
        return instructions
    
    def run_inference(self, text):
        """ Expects formatted text """
        # Run inference
        inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        outputs = self.model(inputs)
        
        # Decode
        logging.info(outputs)
        logging.info(type(outputs))
        logging.info(outputs.shape)
        
        output_ids = torch.argmax(outputs[-1])
        output_ids = self.tokenizer.decode(output_ids)
        logging.info(text)
        logging.info(output_ids)
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        # filter out rows which have no triples
        _data = data.filter(~data['responses'].is_in(['N/A', ""]))
        output = {}
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
                # for each triple, decode the identifiers to labels    
                
                # run inference on the triples' relevance to the question with respect to the expected answer
                prompt = self.get_prompt_triple_relevance(q, a, trip)
                result = self.run_inference(prompt)
                
                # save the output to a dict
                output[row['id']].append(result)
        
        return output
    
    def evaluate(self, data):
        pass