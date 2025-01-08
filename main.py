import src.data.datamanager as dm
import src.utils.config as config
import src.analysis.analyser as anal

def main():
    # Initialize the global configuration
    global_config = config.GlobalConfig()
    global_config.set_random_seeds()
    args = global_config.get_args()

    # Initialize the data manager
    data_manager = dm.DataManager(args)
    anal.DatasetAnalyser(data_manager.df).get_dataset_stats()
    pass

if __name__ == '__main__':
    main()