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
            self.model, self.device = self.get_model(hf_model)
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
    
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
        messages = [
            {"role": "system", "content": r"""You need to evaluate whether a given Wikidata triple (subject-predicate-object) answers a given question and supports the given answers.
            Give me your output in YAML format with of three options, "Yes", "No", "Unsure" and indicate your confidence level.
            The triples can have multiple hops where the object and subject alternates with predicates seperating them.
            
            Here is an expected format of the input:
            Question: What is the capital of France?
            Answer: Paris
            Triple: (France, capital, Paris)
            
            Here is an expected format of the output:
            Relevance: Yes
            Confidence: 0.9x§
            
            Here is another example of expected input:
            Question: Is there gravity on the International Space Station?; 
            Answer: ["Yes, Earth's gravity on the International Space Station is around 90 percent of the gravity on the Earth's surface", 'Yes, there is gravity everywhere in space']; 
            Triple: (International Space Station, topic's main Wikimedia portal, Portal:International Space Station, Wikimedia portal's main topic, International Space Station)
            
            Here is another example of expected output:
            Relevance: No
            Confidence: 0.87
            
            """},
            {"role": "user", "content": f"Question: {question}; \nAnswer: {answer}; \nTriple: ({triple})"},
        ]
        
        # instructions = f"""
        #     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        #     You need to evaluate whether a given Wikidata triple (subject-predicate-object) is informative to the given question with respect to the expected answer.
        #     Output only your answer in YAML format and one of three options, "Yes", "No", "Unsure" and indicate your confidence level.
        #     The triples can have multiple hops where the object and subject alternates with predicates seperating them.
            
        #     Here is an expected format of the input:
        #     Question: What is the capital of France?
        #     Answer: Paris
        #     Triple: (France, capital, Paris)
            
        #     Here is an expected format of the output:
        #     Relevance: Yes
        #     Confidence: 0.9
            
        #     <|eot_id|>
            
        #     <|start_header_id|>user<|end_header_id|>
        #     Question: {question}
        #     Answer: {answer}
        #     Triple: ({", ".join(triple)})
        #     <|eot_id|>
            
        #     <|start_header_id|>assistant<|end_header_id|>
        # """
        
        return messages
    
    def run_inference(self, text):
        """ Expects formatted text """
        # Run inference
        # inputs = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        tokenized_chat = self.tokenizer.apply_chat_template(text, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(
            tokenized_chat, 
            max_new_tokens=256,
            num_beams=5,            # Use beam search for better coherence
            temperature=0.7,        # Low temperature for precise answers
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
    
    def add_labels(self, data: pl.DataFrame):
        _data = data.filter(~data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        # add "trip_labels" column
        _data = _data.with_columns(
                    trip_labels=pl.lit('N/A')
                )
        
        for datapoint in tqdm(_data.iter_rows(named=True), total=_data.shape[0]):
            trips = datapoint.get('responses').split(config.LIST_SEP)
            logging.info(f"Processing row {datapoint['id']} with triples (n={len(trips)})")
            
            labels = []
            for trip in trips:
                if len(trip) == 0: continue
                # for each triple, decode the identifiers to labels    
                _labels = self.kg_manager.decode_statement_labels(trip.split())
                _labels = "; ".join(_labels)
                labels.append(_labels)
            
            labels = f"{config.LIST_SEP}".join(labels)
            datapoint['trip_labels'] = labels
            _datapoint = pl.from_dict(datapoint, strict=False)
            _data = _data.update(_datapoint, on="id")
            _data.write_json(f"{self.args.data_dir}/data_kg_trip_labels.json")
        pass
                
            
        
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        # filter out rows which have no triples
        _data = data.filter(~data['responses'].is_in(['N/A', "", "<NO_PATHS_FOUND>"]))
        df_mappings = {'id': pl.String, 'source_dataset': pl.String, 'domain': pl.String,'input': pl.String, 'output': pl.String, 'trip': pl.String, 'trip_labels': pl.String, 'relevance': pl.String, 'confidence': float}
        output = pl.DataFrame(schema=df_mappings)
        
        relevances = []
        for row in tqdm(_data.iter_rows(named=True)):
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