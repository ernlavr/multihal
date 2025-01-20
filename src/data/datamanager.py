import datasets
import polars as pl
import os
import src.data.datasetloaders as dl
import src.data.column_mapper as cm
import src.analysis.figures as fig
import logging

# TODO:
# - Add configs for including only specific datasets
# - Add configs for including subsets of datasets (e.g. see halueval)

class DataManager():
    def __init__(self, args):
        self.column_mapper = cm.ColumnMapper()
        self.df = self.column_mapper.get_blank_df()
        self.df_pp = self.column_mapper.get_blank_df()
        self.args = args
        self.ds = dl.load_data(args)
        self.merge_data()

        logging.info(f"Parsed dataset of size: {self.df.shape[0]}")
        self.serialize_ds()


    def serialize_ds(self):
        """ Serializes the datasets into the main dataframe """
        data_dir = 'output/data/'
        os.makedirs(data_dir, exist_ok=True)

        if self.args.debug_mode:
            self.df.sample(n=self.args.n_pds).write_csv(os.path.join(data_dir, 'data_sampled.csv'))

        if self.args.save_full_data:
            self.df.write_csv(os.path.join(data_dir, 'multihal_unprocessed.csv'))
        
        if self.args.save_basic_stats:
            fig.plot_ds_stats(self.df)

    
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


        self.df = self.df.with_columns(pl.col('domain').str.to_lowercase())
        # fill in missing values
        self.df = self.df.with_columns(pl.col("domain").replace(None, "N/A"))
        self.df = self.column_mapper.encode_domains(self.df)