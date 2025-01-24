import src.data.datamanager as dm
import src.utils.config as config
import src.analysis.analyser as anlyz
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
    embeddings = None
    analyzer = None

    # Sentence embeddings
    if args.gen_sent_embeds:
        embedder = se.SentenceEmbeddings(data_manager.df, args)
        embeddings = embedder.gen_embeddings(data_manager.df)
        analyzer = anlyz.DatasetAnalyser(embeddings, args)

    if args.remove_duplicates:
        if args.sent_sim_metric == 'bleu':
            similarities = analyzer.get_bleu_scores(embeddings)
            analyzer.remove_duplicates_by_cossim(similarities, threshold=0.7)
        elif args.sent_sim_metric == 'cosine':
            similarities = analyzer.get_cossim(embeddings, 0.9)
            analyzer.remove_duplicates_by_cossim(similarities)
            
        analyzer.log_domains(embeddings)
        analyzer.run_precluster_analysis(embeddings)
        analyzer.run_dim_red(embeddings)

    # Analysis figures
    if args.gen_anlyz_figs:
        fig.Plots(data_manager.df).plot_ds_stats()
    

if __name__ == '__main__':
    main()