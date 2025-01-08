import datasets
import polars as pl
import os
import src.data.datasetloaders as dl
import src.data.column_mapper as cm

# TODO:
# - Add configs for including only specific datasets
# - Add configs for including subsets of datasets (e.g. see halueval)

class DataManager():
    def __init__(self, args):
        self.column_mapper = cm.ColumnMapper()
        self.df = self.column_mapper.get_blank_df()
        self.ds = dl.load_data(args)
        self.merge_data()

        self.serialize_ds(per_ds=5)


    def serialize_ds(self, per_ds: int = None):
        """ Serializes the datasets into the main dataframe """
        os.makedirs('data', exist_ok=True)
        if per_ds is None:
            self.df.write_csv('data/data.csv')
        else:
            sampled_df = pl.DataFrame()
            for dataset_name in self.ds.keys():
                dataset = self.df.filter(self.df['source_dataset'] == dataset_name)
                sampled_dataset = dataset.sample(n=per_ds)
                sampled_df = sampled_df.vstack(sampled_dataset)
            sampled_df.write_csv('data/data_sampled.csv')

    
    def merge_data(self):
        self.df = self.column_mapper.merge_shroom2024(self.df, self.ds['shroom2024']['train'].to_polars(), 'DM')
        self.df = self.column_mapper.merge_shroom2025(self.df, self.ds['shroom2025']['train'].to_polars(), 'DM')
        self.df = self.column_mapper.merge_halueval(self.df, self.ds['halueval']['val'].to_polars())
        self.df = self.column_mapper.merge_truthfulqa_gen(self.df, self.ds['tqa_gen']['val'].to_polars())
        self.df = self.column_mapper.merge_felm(self.df, self.ds['felm']['val'].to_polars())
        self.df = self.column_mapper.merge_halubench(self.df, self.ds['halubench']['val'].to_polars())
        self.df = self.column_mapper.merge_defan(self.df, self.ds['defan']['val'].to_polars())
        self.df = self.column_mapper.merge_simpleqa(self.df, self.ds['simpleqa']['val'].to_polars())
        