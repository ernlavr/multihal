import transformers
import torch

class LLMJudge():
    def __init__(self, hf_model, args):
        self.args = args
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(hf_model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def get_prompt(self):
        """ Returns instruction prompt for the LLM """
        pass
    
    def evaluate(self, data):
        pass