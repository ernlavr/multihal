import datasets
import polars as pl
import os
import src.data.datasetloaders as dl
import src.data.column_mapper as cm
import src.analysis.figures as fig
import logging
import src.utils.singleton as sing
import src.utils.helpers as utils
from src.utils.config import LIST_SEP

# TODO:
# - Add configs for including subsets of datasets (e.g. see halueval)

class DataManager(metaclass=sing.Singleton):
    def __init__(self, args):
        self.column_mapper = cm.ColumnMapper()
        self.df : pl.DataFrame = self.column_mapper.get_blank_df()
        self.df_pp : pl.DataFrame = self.column_mapper.get_blank_df()
        self.args = args
        

        if args.continue_from_previous_state is None and args.load_premade_dataset is None:
            self.ds = dl.load_data(args)
            self.merge_data()
            self.serialize_ds()
        else:
            self.df = self.get_premade_dataset(args)
            
        logging.info(f"Parsed dataset of size: {self.df.shape[0]}")
        
        

    def get_premade_dataset(self, args):
        # get extension
        if args.continue_from_previous_state is not None \
        and args.continue_from_previous_state['dataset'] is not None \
        and args.load_premade_dataset is not None:
            raise ValueError("Ambiguous which dataset to load, previous state and premade are both specified")
        
        if args.continue_from_previous_state is not None:
            ds_path = args.continue_from_previous_state['dataset']
        
        if args.load_premade_dataset is not None:
            ds_path = args.load_premade_dataset
        
        
        ext = ds_path.split('.')[-1]
        if ext == 'csv':
            return pl.read_csv(ds_path)
        elif ext == 'parquet':
            return pl.read_parquet(ds_path)
        elif ext == 'json':
            return pl.read_json(ds_path)

    def get_dataset(self, args=None) -> pl.DataFrame:
        """
        Retrieve a filtered dataset based on the provided arguments.

        Args:
            args (optional): An object containing the dataset filter criteria. 
                     It should have an attribute `datasets` which is a list of dataset names to filter.

        Returns:
            pl.DataFrame: A filtered DataFrame if `args` is provided, otherwise the entire DataFrame.
        """
        if args is not None:
            return self.df.filter(pl.col("source_dataset").is_in(args.datasets))
        return self.df

    def serialize_ds(self):
        """ Serializes the datasets into the main dataframe """
        data_dir = self.args.data_dir
        os.makedirs(data_dir, exist_ok=True)

        if self.args.debug_mode:
            self.df.sample(n=self.args.n_pds).write_csv(os.path.join(data_dir, 'multihal_debug_sampled.csv'))

        if self.args.save_full_data:
            self.df.write_csv(os.path.join(data_dir, 'multihal_unprocessed.csv'))
        
        if self.args.save_basic_stats:
            fig.plot_ds_stats(self.df)

    def subsample(self, n: int, data: pl.DataFrame) -> pl.DataFrame:
        """ Evenly sample the dataframe from each dataset, for each domain/subdomain/subdataset
            to match n as closely and as evenly as possible.
        """
        subsampled = []
        points_per_dataset = n // len(data['source_dataset'].unique())
        
        for ds_name, ds_split in data.group_by('source_dataset', maintain_order=True):
            points_per_subgroup = points_per_dataset // len(ds_split['domain'].unique())
            
            for sub_ds_name, sub_ds_split in ds_split.group_by('domain', maintain_order=True):
                if len(sub_ds_split) < points_per_subgroup:
                    subsampled.append(sub_ds_split)
                else:
                    domain_split_sampled = sub_ds_split.sample(n=points_per_subgroup, seed=self.args.seed)
                    subsampled.append(domain_split_sampled)
        
        output = pl.concat(subsampled)
        output = output.sample(fraction=1.0, with_replacement=False, seed=self.args.seed, shuffle=True)     
        return pl.concat(subsampled)
    
    def remove_refused_answers(self, data: pl.DataFrame) -> pl.DataFrame:
        refusal_strings = [i.lower() for i in utils.get_refusal_strings()]
        tmp = data.with_columns(
            pl.concat_str([
                pl.col("output"), 
                pl.col("optional_output")
                ], 
                separator=LIST_SEP,
                ignore_nulls=True
            ).alias("output_opt_output").str.to_lowercase()
        )    
        mask = tmp["output_opt_output"].str.contains("|".join(refusal_strings))
        data = data.filter(~mask)
        
        logging.info(f"Removed refused LLM answers: {len(mask.filter(mask == True))} refusal answers; New length {len(data)}")
        return data
            
    
    def merge_data(self):
        merge_funcs = {
            'shroom2024': self.column_mapper.merge_shroom2024,
            'shroom2025': self.column_mapper.merge_shroom2025,
            'halueval': self.column_mapper.merge_halueval,
            'tqa_gen': self.column_mapper.merge_truthfulqa_gen,
            'halubench': self.column_mapper.merge_halubench,
            'felm': self.column_mapper.merge_felm,
            'defan': self.column_mapper.merge_defan,
            'simpleqa': self.column_mapper.merge_simpleqa,
        }

        for ds in self.args.datasets:
            # assert len(self.ds[ds].keys()) == 1, "Only one split per dataset is supported"
            for split in list(self.ds[ds].keys()):
                if 'paraphrased' in split:
                    self.df_pp = merge_funcs[ds](self.df_pp, self.ds[ds][split].to_polars())
                else:
                    self.df = merge_funcs[ds](self.df, self.ds[ds][split].to_polars())


        
        # remove where output is None
        self.df = self.df.filter(~pl.col("output").is_null())
        self.df = self.df.with_columns(pl.col('domain').str.to_lowercase())
        # fill in missing values
        self.df = self.df.with_columns(pl.col("domain").replace(None, "N/A"))
        self.df = self.column_mapper.encode_domains(self.df)