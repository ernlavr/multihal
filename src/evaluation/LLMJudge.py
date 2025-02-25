import transformers
import torch
import logging
import src.utils.helpers as utils
import os

class LLMJudge():
    def __init__(self, hf_model, args):
        utils.print_cuda_stats()
        self.args = args
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
    
    def run_inference(self, text):
        # Run inference
        prompt = self.get_instructions() + text
        inputs = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        outputs = self.model(inputs)
        
        # Decode
        logging.info(outputs)
        logging.info(type(outputs))
        logging.info(outputs.shape)
        
        output_ids = torch.argmax(outputs[-1])
        output_ids = self.tokenizer.decode(output_ids)
        logging.info(prompt)
        logging.info(output_ids)
        
    def evaluate_triple_relevance(self, data: pl.DataFrame):
        pass
    
    def evaluate(self, data):
        pass