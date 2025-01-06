import src.data.datamanager as dm
import src.utils.config as config

def main():
    # Initialize the global configuration
    global_config = config.GlobalConfig()
    args = global_config.get_args()

    # Initialize the data manager
    data_manager = dm.DataManager(args)
    pass

if __name__ == '__main__':
    main()