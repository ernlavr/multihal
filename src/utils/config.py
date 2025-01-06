import argparse 
import src.utils.singleton as singl

class GlobalConfig(metaclass=singl.Singleton):
    def __init__(self):
        self._args = self._initialize_args()

    def _initialize_args(self):
        parser = argparse.ArgumentParser(description='MultiHal: Dataset Gathering and Analysis Framework')
        parser.add_argument('--datasets', nargs='+', help="List of datasets to load, available datasets are: shroom2024, shroom2025, halueval, tqa_gen, felm, halubench, defan, simpleqa")
        return parser.parse_args()
    
    def addToGlobalConfig(self, key, value):
        self._args.__dict__[key] = value
        return self.get_args()
    
    def get_args(self):
        return self._args
    