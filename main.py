import src.data.datamanager as dm

import src.utils.config as config
import src.utils.wandb_manager as wbMang

import src.analysis.analyser as anlyz
import src.analysis.figures as fig
import src.analysis.sentence_embedder as se
import src.kgs.kg_manager as kgm
import src.network.udp_manager as br

import src.evaluation.LLMJudge as llmJudge
import logging
import wandb
import pprint

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
            removed = analyzer.remove_duplicates_by_sim_matrix(similarities, threshold=0.9, **args.__dict__)
        elif args.sent_sim_metric == 'cosine':
            similarities = analyzer.get_cossim(embeddings)
            removed = analyzer.remove_duplicates_by_sim_matrix(similarities, threshold=0.99, **args.__dict__)
            
        removed.write_json(f"output/data/multihal_removed_duples_{args.sent_sim_metric}.json")    
        analyzer.log_domains(embeddings)
    if args.run_clustering:
        analyzer.run_precluster_analysis(embeddings)
        analyzer.run_dim_red(embeddings)

    # Analysis figures
    if args.gen_anlyz_figs:
        fig.plot_ds_stats(data_manager.df)
        analyzer = anlyz.DatasetAnalyser(None, None)
        logging.info(pprint.pformat(analyzer.get_list_of_domains_and_counts(data_manager.df)))        
    
    # Fetch KG triples for QA section of the dataset
    if args.parse_text_to_ents:
        ds = data_manager.get_dataset(args)
        # shuffle
        ds_shuffled = ds.sample(len(ds), shuffle=True)
        kg_manager = kgm.KGManager(ds_shuffled)
        kg_manager.process(ds_shuffled)
    
    if args.run_qa_kgs:
        logging.info("Starting querying KGs")
        ds = data_manager.get_dataset(args)
        bridge = br.NetworkBridge()
        kg_manager = kgm.KGManager(ds)
        kg_manager.query_kg(ds, bridge, max_hops=3)
        logging.info("Finished querying KGs")
        
    if args.evaluate:
        # judge = llmJudge.LLMJudge('meta-llama/Llama-3.3-70B-Instruct', args)
        judge = llmJudge.LLMJudge(args.llm_judge_model, args)
        judge.evaluate_triple_relevance(data_manager.df)
        
    

if __name__ == '__main__':
    logging.info("Starting main")
    main()