import polars as pl
from datasets import Dataset

class ColumnMapper():
    def __init__(self):
        """ Defines column names of MultiHal and mappings of source datasets to MultiHal columns """
        self.ID = 'id'   # identifier to the data point
        self.SOURCE_DATASET = 'source_dataset' # dataset from which the data point was sourced
        self.TASK = 'task' # task of the data point
        self.DOMAIN = 'domain' # domain of the data point
        self.INPUT = 'input' # input to the model
        self.OUTPUT = 'output' # correct output
        self.OPTIONAL_OUTPUT = 'optional_output' # optional outputs
        self.INCORRECT_ANSWERS = 'incorrect_answers' # incorrect outputs
        self.CONTEXT = 'context' # context for the data point
        self.CONTEXT_TYPE = 'context_type' # type of context, web or passage

    def get_blank_df(self):
        return pl.DataFrame([], self.get_multihal_columns())

    def get_multihal_columns(self):
        return {
            self.ID: str,
            self.SOURCE_DATASET: str,
            self.TASK : str,
            self.DOMAIN: str,
            self.INPUT: str,
            self.OUTPUT: str,
            self.OPTIONAL_OUTPUT: str,
            self.INCORRECT_ANSWERS: str,
            self.CONTEXT: str,
            self.CONTEXT_TYPE: str,
        }
        
    def get_shroom2024_mappings(self, tgt):
        """ Mappings for the SHROOM2024 dataset.
            tgt: Denotes whether reference is hyp or ref
        """
        cols = {'model', 'src', 'task', 'hyp', 'ref', 'tgt'}
        mappings = {
            cols['src']: self.INPUT,
            cols['ref']: self.OUTPUT,
        }
        return mappings

    def get_shroom2025_mappings(self):
        cols = ['lang', 'model_id', 'model_input', 'model_output_text', 'model_output_logits', 'model_output_tokens']
        mappings = {
            cols['model_input']: self.INPUT,
            cols['model_output_text']: self.OUTPUT,
        }
        return mappings

    def get_halueval_mappings(self):
        cols = ['knowledge', 'question', 'right_answer', 'hallucinated_answer']
        mappings = {
            cols['question']: self.INPUT,
            cols['right_answer']: self.OUTPUT,
            cols['knowledge']: self.CONTEXT,
        }
        return mappings

    def get_tqa_gen_mappings(self):
        """ Type denotes adversarial/non-adversarial

        """
        cols = ['type', 'category', 'question', 'best_answer', 'correct_answers', 'incorrect_answers', 'source']
        mappings = {
            cols['category']: self.DOMAIN,
            cols['question']: self.INPUT,
            cols['best_answer']: self.OUTPUT,
            cols['correct_answers']: self.OPTIONAL_OUTPUT,
            cols['incorrect_answers']: self.INCORRECT_ANSWERS,
            cols['source']: self.CONTEXT,
        }
        return mappings

    def get_felm_mappings(self):
        cols = ['index', 'prompt', 'response', 'segmented_response', 'labels', 'comment', 'type', 'ref', 'source']
        mappings = {
            cols['index']: self.ID,
            cols['prompt']: self.INPUT,
            cols['response']: self.OUTPUT,
            cols['ref']: self.CONTEXT,
        }

    def get_halubench_mappings(self):
        cols = ['id', 'passage', 'question', 'answer', 'label', 'source_ds']
        mappings = {
            cols['id']: self.ID,
            cols['question']: self.INPUT,
            cols['answer']: self.OUTPUT,
            cols['passage']: self.CONTEXT,
        }

    def defan_mappings(self):
        cols = ['questions', 'answer', 'type']
        mappings = {
            cols['questions']: self.INPUT,
            cols['answer']: self.OUTPUT,
        }

    def get_simpleqa_mappings(self):
        cols = ['metadata', 'problem', 'answer']
        mappings = {
            cols['problem']: self.INPUT,
            cols['answer']: self.OUTPUT,
        }

    def map_shroom2024(self, multihal: pl.DataFrame, shroom2024: pl.DataFrame):
        mappings = self.get_shroom2024_mappings(tgt)
        return df.select([mappings[col] for col in df.columns])