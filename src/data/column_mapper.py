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
        
    def get_shroom2024_mappings(self):
        """ Mappings for the SHROOM2024 dataset.
            tgt: Denotes whether reference is hyp or ref
        """
        cols = ['model', 'src', 'task', 'hyp', 'ref', 'tgt']
        mappings = {
            'src': self.INPUT,
            'ref': self.OUTPUT,
        }
        return mappings

    def get_shroom2025_mappings(self):
        cols = ['lang', 'model_id', 'model_input', 'model_output_text', 'model_output_logits', 'model_output_tokens']
        mappings = {
            'model_input': self.INPUT,
            'model_output_text': self.OUTPUT,
        }
        return mappings

    def get_halueval_mappings(self):
        cols = ['knowledge', 'question', 'right_answer', 'hallucinated_answer']
        mappings = {
            'question': self.INPUT,
            'right_answer': self.OUTPUT,
            'knowledge': self.CONTEXT,
        }
        return mappings

    def get_tqa_gen_mappings(self):
        """ Type denotes adversarial/non-adversarial

        """
        cols = ['type', 'category', 'question', 'best_answer', 'correct_answers', 'incorrect_answers', 'source']
        mappings = {
            'category': self.DOMAIN,
            'question': self.INPUT,
            'best_answer': self.OUTPUT,
            'correct_answers': self.OPTIONAL_OUTPUT,
            'incorrect_answers': self.INCORRECT_ANSWERS,
            'source': self.CONTEXT,
        }
        return mappings

    def get_felm_mappings(self):
        cols = ['index', 'prompt', 'response', 'segmented_response', 'labels', 'comment', 'type', 'ref', 'source']
        mappings = {
            'index': self.ID,
            'prompt': self.INPUT,
            'response': self.OUTPUT,
            'ref': self.CONTEXT,
        }
        return mappings

    def get_halubench_mappings(self):
        cols = ['id', 'passage', 'question', 'answer', 'label', 'source_ds']
        mappings = {
            'id': self.ID,
            'question': self.INPUT,
            'answer': self.OUTPUT,
            'passage': self.CONTEXT,
        }
        return mappings

    def defan_mappings(self):
        cols = ['questions', 'answer', 'type']
        mappings = {
            'questions': self.INPUT,
            'answer': self.OUTPUT,
        }
        return mappings

    def get_simpleqa_mappings(self):
        cols = ['metadata', 'problem', 'answer']
        mappings = {
            'problem': self.INPUT,
            'answer': self.OUTPUT,
        }
        return mappings

    def merge_dataframes(self, primary: pl.DataFrame, secondary: pl.DataFrame, mappings: dict) -> pl.DataFrame:
        """
        Merge values from primary into secondary based on column mappings

        Args:
            primary: Source dataframe
            secondary: Target dataframe
            mappings: Dict mapping primary columns to secondary columns
        """
        # Rename columns in secondary dataframe based on mappings
        secondary = secondary.rename(mappings)
        
        # Add missing columns to secondary dataframe with null values
        for col in primary.columns:
            if col not in secondary.columns:
                secondary = secondary.with_columns(pl.lit(None).alias(col))
        
        # Reorder columns in secondary dataframe to match primary dataframe
        secondary = secondary.select(primary.columns)
        
        # Concatenate primary and secondary dataframes
        merged = pl.concat([primary, secondary], how="vertical")
        return merged

        

    def map_shroom2024(self, multihal: pl.DataFrame, shroom2024: pl.DataFrame):
        mappings = self.get_shroom2024_mappings()
        merged = self.merge_dataframes(multihal, shroom2024, mappings)
        return merged