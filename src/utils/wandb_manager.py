import src.utils.singleton as singl
import os
import wandb
import polars as pl

class WandbManager(metaclass=singl.Singleton):
    def __init__(self, args):
        self._args = args
        self.init_wandb()

    def get_args(self):
        return self._args

    def init_wandb(self):
        if self._args.wandb_online:
            os.environ["WANDB_MODE"] = "online"
        
        wandb.init(project="multihal", config=self._args.__dict__)

    def log_dataframe(self, data: pl.DataFrame):
        # Remove embeddings column for clarity
        data = data.drop('embeddings')
        table = wandb.Table(dataframe=data.to_pandas())
        wandb.log({'Table': table})


    def log_figure(self, figure_name, fig):
        run = wandb.run
        if run is None:
            return
        
        img = wandb.Image(fig)
        run.log({figure_name: img})