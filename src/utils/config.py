import argparse 
import src.utils.singleton as singl
import src.utils.logger as logger
import torch
import random
import numpy as np
from types import SimpleNamespace
import argparse
import logging
import omegaconf 
import torch
from dotenv import load_dotenv
import wandb
import os
import pprint

def init_wandb(args):
    if args.wandb_online:
        os.environ["WANDB_MODE"] = "online"
        wandb.init(project="multihal", config=args.__dict__)

# Constants
LIST_SEP = ' <SEP> '

class GlobalConfig(metaclass=singl.Singleton):
    def __init__(self):
        self._args = self._initialize_args()
        logger.KgLogger(create_log=True)
        logging.info(f"Global configuration: {pprint.pformat(vars(self._args))}")


    def _initialize_args(self):
        load_dotenv('.env')
        config_args = self.get_config_args()
        override_args = self.load_yaml(config_args)
        override_args.wandb_online = config_args.wandb_online
        
        default_args = self.get_default_args()
        return self.override_args(default_args, override_args)
    
    def addToGlobalConfig(self, key, value):
        self._args.__dict__[key] = value
        return self.get_args()
    
    def get_args(self):
        return self._args
    
    def set_random_seeds(self, seed=42):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def get_config_args(self) -> dict:
        """
        Parses command line arguments related to configuration for the application.

        The function supports the following arguments:
        --wandb: A flag indicating whether to use Weights and Biases in online mode for experiment tracking. Default is False.
        --config: The path to the configuration file. Default is None.

        Returns:
            An object that contains the values of the command line arguments.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--wandb-online",
            action="store_true",
            help="Use Weights and Biases in online mode for experiment tracking",
            default=False,
        )
        parser.add_argument("--config", type=str, help="Path to config file", default=None)

        return parser.parse_args()    
    

    def get_default_args(self) -> dict:
        """ Return a dict object of default argument mappings. This is useful for
        working with

        Returns:
            dict: _description_
        """
        return dict(
            # General
            num_workers=4,
            seed=42,    # seed for all random-based function calls
            log_level="INFO",
            device="cuda" if torch.cuda.is_available() else "cpu",
            SLRUM_JOB_ID=os.getenv("SLURM_JOB_ID"),

            # Data
            data_dir="data",
            datasets="all", # all, shroom2024, shroom2025, halueval, tqa_gen, felm, halubench, defan, simpleqa
            save_full_data=False,
            
            # Control
            debug_mode=True,                # Debug mode
            n_pds=5,                        # Number of datapoints to sample for debugging
            gen_anlyz_figs=False,
            gen_sent_embeds=False,
            wandb_online=False,
            sent_sim_metric=None,
            remove_duplicates=False,

            # clustering
            clustering_algo=None,
            run_clustering=False,

            # Model
            sentence_embedder="sentence-transformers/sentence-t5-base", # HuggingFace or local path
            )

    def load_yaml(self, config_args: dict) -> dict:
        """
        Load a YAML file and return its contents as a dictionary.

        Args:
            path (str): The path to the YAML file.

        Returns:
            dict: The contents of the YAML file.
        """
        return omegaconf.OmegaConf.load(config_args.config)
    
    def override_args(self, default_args : dict, override_args : dict) -> dict:
        """
        Override the default arguments with the provided override arguments.

        Args:
            default_args (dict): The default arguments.
            override_args (dict): The override arguments.

        Returns:
            dict: The merged arguments.
        """
        default_args.update(override_args)
        return SimpleNamespace(**default_args)