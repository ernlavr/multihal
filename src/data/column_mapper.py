import polars as pl
from datasets import Dataset
from src.utils.singleton import Singleton
import src.utils.config as config

class ColumnMapper(metaclass=Singleton):
    def __init__(self):
        """ Defines column names of MultiHal and mappings of source datasets to MultiHal columns """
        self.ID = 'id'   # identifier to the data point
        self.SOURCE_DATASET = 'source_dataset' # dataset from which the data point was sourced
        self.TASK = 'task' # task of the data point
        self.SPLIT = 'split' # split of the data point
        self.DOMAIN = 'domain' # domain of the data point
        self.INPUT = 'input' # input to the model
        self.OUTPUT = 'output' # correct output
        self.OPTIONAL_OUTPUT = 'optional_output' # optional outputs
        self.INCORRECT_ANSWERS = 'incorrect_answers' # incorrect outputs
        self.CONTEXT = 'context' # context for the data point
        self.CONTEXT_TYPE = 'context_type' # type of context, web or passage

    def get_blank_df(self):
        return pl.DataFrame([], self.get_multihal_columns())
    
    def encode_domains(self, data: pl.DataFrame):
        return data.with_columns(
            pl.col('domain')
            .rank('dense')
            .cast(pl.Int64).name.suffix('_encoded')
        )
    
    def get_domain_mappings(self):
        domains = {

        }
        return domains

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
        
    def get_shroom2024_mappings(self, task):
        """ Mappings for the SHROOM2024 dataset.
            tgt: Denotes whether reference is hyp or ref
        """
        cols = ['model', 'src', 'task', 'hyp', 'ref', 'tgt']
        mappings = {
            'src': self.INPUT,
        }

        if task == 'PG':
            mappings['src'] = self.OUTPUT
        elif task == 'DM':
            mappings['tgt'] = self.OUTPUT
        else:
            raise ValueError(f"Task {task} not recognized for Shroom 2024")

        return mappings

    def get_shroom2025_mappings(self):
        cols = ['lang', 'model_id', 'model_input', 'model_output_text', 'model_output_logits', 'model_output_tokens']
        mappings = {
            'model_input': self.INPUT,
            'model_output_text': self.OUTPUT,
            'task': self.TASK,
        }
        return mappings

    def get_halueval_mappings(self):
        cols = ['knowledge', 'question', 'right_answer', 'hallucinated_answer']
        mappings = {
            'question': self.INPUT,
            'right_answer': self.OUTPUT,
            'knowledge': self.CONTEXT,
            'task': self.TASK,
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
            'task': self.TASK,
        }
        return mappings

    def get_felm_mappings(self):
        cols = ['index', 'prompt', 'response', 'segmented_response', 'labels', 'comment', 'type', 'ref', 'source']
        mappings = {
            'prompt': self.INPUT,
            'response': self.OUTPUT,
            'ref': self.CONTEXT,
            'task': self.TASK,
            'domain': self.DOMAIN,
        }
        return mappings

    def get_halubench_mappings(self):
        cols = ['id', 'passage', 'question', 'answer', 'label', 'source_ds']
        mappings = {
            'id': self.ID,
            'passage': self.CONTEXT,
            'question': self.INPUT,
            'answer': self.OUTPUT,
        }
        return mappings

    def defan_mappings(self):
        cols = ['questions', 'answer', 'type', 'domain']
        mappings = {
            'questions': self.INPUT,
            'answer': self.OUTPUT,
            'domain': self.DOMAIN,
            'task': self.TASK,
        }

        return mappings

    def get_simpleqa_mappings(self):
        cols = ['metadata', 'problem', 'answer']
        mappings = {
            'problem': self.INPUT,
            'answer': self.OUTPUT,
            'task': self.TASK,
            'context': self.CONTEXT,
            'domain': self.DOMAIN,
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
    
    def merge_df(self, primary: pl.DataFrame, secondary: pl.DataFrame, mappings: list) -> pl.DataFrame:
        """
        Merge values from primary into secondary based on column mappings

        Args:
            primary: Source dataframe
            secondary: Target dataframe
            mappings: List of dicts mapping primary columns to secondary columns
        """
        # Rename columns in secondary dataframe based on mappings
        for mapping in mappings:
            secondary = secondary.rename(mapping)
        
        # Add missing columns to secondary dataframe with null values
        for col in primary.columns:
            if col not in secondary.columns:
                secondary = secondary.with_columns(pl.lit(None).alias(col))
        
        # Reorder columns in secondary dataframe to match primary dataframe
        secondary = secondary.select(primary.columns)
        
        # Concatenate primary and secondary dataframes
        merged = pl.concat([primary, secondary], how="vertical")
        return merged

    def add_metadata(self, primary: pl.DataFrame, secondary: str, src_ds) -> pl.DataFrame:
        last_merge_len = len(secondary)
        last_merge = primary[-last_merge_len:]

        # is source_dataset present?
        if last_merge['source_dataset'].is_null().all():
            last_merge = last_merge.with_columns(pl.col('source_dataset').fill_null(src_ds))

        # are IDs present?
        if last_merge['id'].is_null().all():
            last_merge = last_merge.with_columns(pl.Series("id", [f"{src_ds}_" + str(i) for i in range(0, len(last_merge))]))

        return pl.concat([primary[:-last_merge_len], last_merge], how='vertical')
    
    def convert_list_str_to_str(self, ds: pl.DataFrame) -> pl.DataFrame:
        """ Converts columns with list type to string type """
        list_columns = [col for col in ds.columns if ds[col].dtype == pl.datatypes.List(pl.datatypes.String)]
        for col in list_columns:
            ds.replace_column(ds.get_column_index(col), ds[col].list.join(config.LIST_SEP))
        return ds    

    def merge_shroom2024(self, multihal: pl.DataFrame, shroom2024: pl.DataFrame, task='DM') -> pl.DataFrame:
        if task != 'all':
            shroom2024 = shroom2024.filter(pl.col('task') == task)
        mappings = self.get_shroom2024_mappings(task)
        merged = self.merge_dataframes(multihal, shroom2024, mappings)
        merged = self.add_metadata(merged, shroom2024, 'shroom2024')
        return merged
    
    def merge_shroom2025(self, multihal: pl.DataFrame, shroom2025: pl.DataFrame, task='DM') -> pl.DataFrame:
        mappings = self.get_shroom2025_mappings()
        merged = self.merge_dataframes(multihal, shroom2025, mappings)
        merged = self.add_metadata(merged, shroom2025, 'shroom2025')
        return merged
    
    def merge_halueval(self, multihal: pl.DataFrame, halueval: pl.DataFrame, task=None) -> pl.DataFrame:
        mappings = self.get_halueval_mappings()
        merged = self.merge_dataframes(multihal, halueval, mappings)
        merged = self.add_metadata(merged, halueval, 'halueval')
        return merged

    def merge_truthfulqa_gen(self, multihal: pl.DataFrame, tqa_gen: pl.DataFrame, task=None) -> pl.DataFrame:
        mappings = self.get_tqa_gen_mappings()
        # Preprocess tqa_gen, it contains lists of strings so we need to convert them to strings  
        tqa_gen = self.convert_list_str_to_str(tqa_gen)  

        merged = self.merge_dataframes(multihal, tqa_gen, mappings)
        merged = self.add_metadata(merged, tqa_gen, 'tqa_gen')
        return merged
    
    def merge_felm(self, multihal: pl.DataFrame, felm: pl.DataFrame, task=None) -> pl.DataFrame:
        mappings = self.get_felm_mappings()
        felm = self.convert_list_str_to_str(felm)

        # select only datapoints with all "true" labels
        felm = felm.filter(pl.col('labels').list.all())

        merged = self.merge_dataframes(multihal, felm, mappings)
        merged = self.add_metadata(merged, felm, 'felm')
        return merged
    
    def merge_halubench(self, multihal: pl.DataFrame, halubench: pl.DataFrame, task=None) -> pl.DataFrame:
        """ Filter out FAIL labels, so we have only truthful answers """
        mappings = self.get_halubench_mappings()

        # remove rows from halubench where 'source_ds' is 'halubench'
        halubench = halubench.filter(pl.col('source_ds') != 'halueval')

        # remove rows from halubench where 'label' is not 'FAIL'
        halubench = halubench.filter(pl.col('label') == 'PASS')

        # map source datasets to domains
        halubench_domains = {'DROP': 'general', 'covidQA': 'covid', 'pubmedQA': 'pubmed', 'FinanceBench': 'finance', 'RAGTruth': 'ragtruth'}
        
        halubench = halubench.with_columns(
            pl.col('source_ds') \
                .replace(halubench_domains) \
                .alias('domain')
        )

        merged = self.merge_dataframes(multihal, halubench, mappings)
        merged = self.add_metadata(merged, halubench, 'halubench')

        return merged
    
    def merge_defan(self, multihal: pl.DataFrame, defan: pl.DataFrame, task=None) -> pl.DataFrame:
        mappings = self.defan_mappings()
        merged = self.merge_dataframes(multihal, defan, mappings)
        merged = self.add_metadata(merged, defan, 'defan')
        return merged
    
    def merge_simpleqa(self, multihal: pl.DataFrame, simpleqa: pl.DataFrame, task=None) -> pl.DataFrame:
        mappings = self.get_simpleqa_mappings()
        merged = self.merge_dataframes(multihal, simpleqa, mappings)
        merged = self.add_metadata(merged, simpleqa, 'simpleqa')
        return merged