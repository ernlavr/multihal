import src.data.datamanager as dm
import src.utils.config as config
import src.analysis.analyser as anal
import src.analysis.figures as fig
import src.analysis.sentence_embedder as se
import src.utils.wandb_manager as wbMang
import logging
import wandb

def main():
    # Initialize the global configuration
    global_config = config.GlobalConfig()
    global_config.set_random_seeds()
    args = global_config.get_args()
    config.init_wandb(args)
    wbMang.WandbManager(args)

    # Initialize the data manager
    logging.info("Starting data manager")
    data_manager = dm.DataManager(args)

    # Sentence embeddings
    if args.gen_sent_embeds:
        embedder = se.SentenceEmbeddings(data_manager.df, args)
        embeddings = embedder.gen_embeddings(data_manager.df)
        analyzer = anal.DatasetAnalyser(embeddings, args)
        analyzer.remove_duplicates_by_cossim(embeddings)
        # analyzer.get_sentence_similarities(embeddings, 0.9)
        analyzer.run_precluster_analysis(embeddings)
        analyzer.run_dim_red(embeddings)

    # Analysis figures
    if args.gen_anal_figs:
        fig.Plots(data_manager.df).plot_ds_stats()
    

if __name__ == '__main__':
    main()