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
        
        self.serialize_ds()


    def serialize_ds(self):
        """ Serializes the datasets into the main dataframe """
        output = pl.DataFrame([], {'source': str})
        for ds_name, ds in self.ds.items():
            for subset in ds.keys():
                df = ds[subset].to_polars()
                head = df.head(1)

                head = head.rename(lambda col_name : ds_name + '_' + col_name)


                # add the dataset by new columns
                output = pl.concat([output, head], how='diagonal_relaxed')
        
        output.write_json('output.json')
        return output