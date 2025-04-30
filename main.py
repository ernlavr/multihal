"""
Module: main_pipeline.py

This module contains functions for processing, analyzing, and evaluating datasets.
It performs tasks such as generating sentence embeddings, removing duplicate datapoints,
subsampling the dataset, clustering, generating analysis figures, processing knowledge graphs (KGs),
querying KGs, and evaluating triples using an LLM-based judge.

The main() function sets up the configuration, logging, and external integrations (e.g., wandb)
before running the defined processing steps based on the provided arguments.
"""

import logging
import pprint
from typing import Any, Tuple
import polars as pl

import numpy as np
import wandb

import src.data.datamanager as dm
import src.data.paraphraser as pp
import src.utils.config as config
import src.utils.wandb_manager as wbMang
import src.analysis.analyser as anlyz
import src.analysis.figures as fig
import src.analysis.sentence_embedder as se
import src.kgs.kg_manager as kgm
import src.network.udp_manager as br
import src.evaluation.LLMJudge as llmJudge
import src.translation.translator as tl

import src.evaluation.Api_judge as apiJudge
import src.evaluation.KnowledgeInjection as ki
import src.utils.constants as const

# import src.evaluation.DeepEval as deJudge



def generate_sentence_embeddings(dataset: pl.DataFrame, args: Any) -> Tuple[np.ndarray, anlyz.DatasetAnalyser]:
    """
    Generate sentence embeddings from the dataset and create a corresponding analyzer.

    Parameters:
        data_manager (dm.DataManager): The data manager holding the dataset.
        args (Any): Command-line arguments or configuration object.

    Returns:
        Tuple[np.ndarray, anlyz.DatasetAnalyser]: A tuple containing the embeddings and
                                                   the dataset analyzer.
    """
    # Create the embedder using the dataset and args
    embedder = se.SentenceEmbeddings(dataset, args)
    # Generate embeddings from the dataset
    embeddings = embedder.gen_embeddings(dataset)
    # Initialize the analyzer with the generated embeddings
    analyzer = anlyz.DatasetAnalyser(dataset, args)
    logging.info(f"Number of datapoints after embedding: {embeddings.shape[0]}")
    return embeddings, analyzer


def remove_duplicates(args: Any, dataframe: Any, analyzer: anlyz.DatasetAnalyser) -> Any:
    """
    Remove duplicate entries from the dataset based on a similarity metric.

    Parameters:
        args (Any): Configuration object containing the similarity metric and thresholds.
        dataframe (Any): The dataset (or embeddings) to check for duplicates.
        analyzer (anlyz.DatasetAnalyser): The analyzer used to compute similarity scores.

    Returns:
        Any: The dataset with duplicates removed.
    """
    # Determine similarity computation and threshold based on the selected metric
    if args.sent_sim_metric == 'bleu':
        similarities = analyzer.get_bleu_scores(dataframe)
        threshold = 0.9
    elif args.sent_sim_metric == 'cosine':
        similarities = analyzer.get_cossim(dataframe)
        threshold = 0.99
    else:
        raise ValueError(f"Unknown similarity metric: {args.sent_sim_metric}")

    # Remove duplicates based on the similarity matrix and provided threshold.
    removed = analyzer.remove_duplicates_by_sim_matrix(similarities, dataframe, threshold=threshold, **args.__dict__)
    # Save the removed duplicates information to a JSON file
    removed.write_json(f"{args.data_dir}/multihal_removed_duples_{args.sent_sim_metric}.json")
    # Log domain distribution after duplicate removal
    analyzer.log_domains(dataframe)
    logging.info(f"Number of datapoints after removing duplicates: {removed.shape[0]}")
    return removed


def subsample_data(data_manager: dm.DataManager, dataset: pl.DataFrame, args: Any) -> Any:
    """
    Subsample the dataset and generate corresponding pie charts for dataset and domain counts.

    Parameters:
        data_manager (dm.DataManager): The data manager with access to the dataset.
        args (Any): Configuration object containing subsample size and figure directory.

    Returns:
        Any: The subsampled dataset.
    """
    # Subsample the dataset using the provided sample size
    dataset = data_manager.subsample(args.subset_sample_size, dataset)
    # Plot dataset counts by source
    labels, counts = np.unique(dataset['source_dataset'], return_counts=True)
    fig.plot_pie({"Dataset counts": (labels, counts)}, "Dataset counts", args.fig_dir + "/subsampling")

    # For each source dataset, plot domain counts
    for source_ds, ds_grouped in dataset.group_by('source_dataset'):
        domain_labels, domain_counts = np.unique(ds_grouped['domain'], return_counts=True)
        fig.plot_pie({"Domain counts": (domain_labels, domain_counts)},
                     f"Domain counts for {source_ds}", args.fig_dir + "/subsampling")

    logging.info(f"Number of datapoints after subsampling: {dataset.shape[0]}")
    return dataset


def run_clustering(analyzer: anlyz.DatasetAnalyser, embeddings: np.ndarray) -> None:
    """
    Run pre-clustering analysis and dimensionality reduction on the embeddings.

    Parameters:
        analyzer (anlyz.DatasetAnalyser): The analyzer with clustering methods.
        embeddings (np.ndarray): The sentence embeddings to cluster.
    """
    analyzer.run_precluster_analysis(embeddings)
    analyzer.run_dim_red(embeddings)


def generate_analysis_figures(dataset: pl.DataFrame, args: Any) -> None:
    """
    Generate analysis figures and log domain statistics.

    Parameters:
        data_manager (dm.DataManager): The data manager containing the dataset.
        args (Any): Configuration object.
    """
    # Plot dataset statistics
    fig.plot_ds_stats(dataset)
    # Initialize a temporary analyzer to get domain statistics
    analyzer = anlyz.DatasetAnalyser(None, None)
    domain_stats = analyzer.get_list_of_domains_and_counts(dataset)
    logging.info(pprint.pformat(domain_stats))
    number_of_dp_with_wiki = analyzer.get_number_of_dp_with_wiki_in_context(dataset)
    if len(number_of_dp_with_wiki) > 0:
        ans_types = number_of_dp_with_wiki['answer_type'].value_counts()
        ans_types = list(zip(ans_types['answer_type'].to_list(), ans_types['count'].to_list()))
        logging.info(f"Number of datapoints with wiki against ans type: {ans_types}")


def process_kg(dataset: Any, args: Any) -> Any:
    """
    Process the knowledge graph (KG) for the given dataset.

    Parameters:
        dataset (Any): The dataset to process.
        args (Any): Configuration object.

    Returns:
        Any: The processed dataset after KG processing.
    """
    kg_manager = kgm.KGManager(dataset, args)
    return kg_manager.process(dataset)


def query_kg(dataset: Any, args: Any) -> None:
    """
    Query the knowledge graph using a network bridge.

    Parameters:
        dataset (Any): The dataset to query.
        args (Any): Configuration object.
    """
    bridge = br.NetworkBridge()
    kg_manager = kgm.KGManager(dataset, args)
    # Query the knowledge graph with a maximum of 3 hops
    kg_manager.query_kg(dataset, bridge, max_hops=2)
    logging.info("Finished querying KGs")

def filter_paths(dataset: Any, args: Any) -> None:
    """
    Filter paths in the dataset based on the specified criteria.

    Parameters:
        dataset (Any): The dataset to filter.
        args (Any): Configuration object.
    """
    kg_manager = kgm.KGManager(dataset, args)
    # Filter paths based on the provided criteria
    dataset = kg_manager.filter_paths(dataset)
    # Save the filtered paths to a JSON file
    dataset.write_json(f"{args.data_dir}/filtered_paths.json")
    logging.info("Finished filtering paths")
    return dataset

def get_trip_labels(dataset: Any, args: Any) -> None:
    """
    Get the triplet labels for the dataset.

    Parameters:
        dataset (Any): The dataset to process.
        args (Any): Configuration object.
    """
    kg_manager = kgm.KGManager(dataset, args)
    # Get triplet labels from the knowledge graph
    dataset = kg_manager.add_labels(dataset)
    logging.info("Finished getting trip labels")
    return dataset

def evaluate_triples(dataset: Any, args: Any) -> None:
    """
    Evaluate the relevance of triples in the dataset using an LLM judge and plot the relevance counts.

    Parameters:
        dataset (Any): The dataset containing triples.
        args (Any): Configuration object containing LLM parameters.
    """
    if args.llm_judge_method == 'proprietary':
        judge = llmJudge.LLMJudge(args.llm_judge_model, args)
    elif args.llm_judge_method == 'deepeval':
        judge = deJudge.DeepEvalJudge(args.llm_judge_model, dataset, args)
    elif args.llm_judge_method == 'api':
        judge = apiJudge.API_Judge(args.llm_judge_model, args)
    else:
        raise ValueError(f"Unknown LLM judge method: {args.llm_judge_method}")
    
    if args.select_labels:
        dataset = judge.choose_best_triples(dataset)
    
    if args.rank_labels:
        dataset = judge.evaluate_triple_relevance(dataset)
    
    return dataset
    
    # # judge.add_labels(dataset)
    # outputs, relevances = judge.evaluate_triple_relevance(dataset)
    # # Compute unique relevance values and their counts
    # relevance_values, relevance_counts = np.unique(relevances, return_counts=True)
    # fig.plot_pie({"Relevance count": (relevance_values, relevance_counts)},
    #              "LLM as judge relevance counts;")
    
def translate(dataset: Any, dataset_pp, args: Any) -> None:
    """
    Translate the dataset using a specified LLM model.

    Parameters:
        dataset (Any): The dataset to translate.
        args (Any): Configuration object containing translation parameters.
    """
    translator = tl.Translator(args.llm_translation_model, args)
    # Translate the dataset
    df = translator.translate_df(dataset, cols=['input', 'output'])
    df_pp = translator.translate_df(dataset_pp, cols=['input'])
    
    # Save the translated dataset to a JSON file
    df.write_json(f"{args.data_dir}/translated_dataset.json")
    df_pp.write_json(f"{args.data_dir}/translated_dataset_pp.json")
    
    logging.info("Finished translating dataset")
    return dataset

def previous_state_continuations(dataset: pl.DataFrame, args) -> pl.DataFrame:
    # Load the dataset to continue from
    state = args.continue_from_previous_state
    dataset_loc = state.get("dataset")
    RUN_DIR = state.get("RUN_DIR")
    function_name = state.get("functions")
    
    # if any of them are null, throw exception
    if dataset_loc is None or RUN_DIR is None or function_name is None:
        raise ValueError("Previous state is not properly defined")
    
    # Run the functions
    for i in function_name:
        func = globals()[i]
        dataset = func(dataset, args)
    

def main() -> None:
    """
    Main pipeline execution. Reads configuration, sets random seeds, initializes external integrations,
    and sequentially executes the data processing, analysis, KG processing, and evaluation steps.
    """
    # Initialize global configuration and get arguments
    global_config = config.GlobalConfig()
    args = global_config.get_args()
    global_config.set_random_seeds(args.seed)
    config.init_wandb(args)
    wbMang.WandbManager(args)

    logging.info("Starting data manager")
    data_manager = dm.DataManager(args)
    analyzer = None
    dataset = data_manager.get_dataset(args)
    dataset = data_manager.cleanup(dataset)
    dataset_pp = None
    logging.info(f"Dataset length: {dataset.shape[0]}")
    
    # DEBUG: get only datapoints with answer_type == "date"
    # dataset = dataset.filter(pl.col("answer_type").is_in([const.ANS_TYPE_DATE, const.ANS_TYPE_NUMBER, const.ANS_TYPE_RANK]))
    
    if args.continue_from_previous_state:
        previous_state_continuations(dataset, args)
        return
    
    # Subsample the dataset if a sample size is provided
    if args.subset_sample_size is not None:
        dataset = subsample_data(data_manager, dataset, args)
    
    if args.remove_refused_answers:
        dataset = data_manager.remove_refused_answers(dataset)
        
    # Generate sentence embeddings if enabled
    if args.gen_sent_embeds:
        dataset, analyzer = generate_sentence_embeddings(dataset, args)

    # Remove duplicate entries if enabled
    if args.remove_duplicates:
        dataset = remove_duplicates(args, dataset, analyzer)

    # Run clustering analysis if enabled
    if args.run_clustering:
        run_clustering(analyzer, dataset)

    # Generate analysis figures if enabled
    if args.gen_anlyz_figs:
        generate_analysis_figures(dataset, args)

    # Process knowledge graphs from text entities if enabled
    if args.parse_text_to_ents:
        dataset = process_kg(dataset, args)

    # Query knowledge graphs if enabled
    if args.run_qa_kgs:
        dataset = query_kg(dataset, args)

    if args.filter_paths:
        dataset = filter_paths(dataset, args)

    if args.get_trip_labels:
        dataset = get_trip_labels(dataset, args)

    # Evaluate triples using the LLM judge if enabled
    if args.select_labels:
        dataset = evaluate_triples(dataset, args)
    
    if args.rank_labels:
        dataset = evaluate_triples(dataset, args)
    
    if args.generate_paraphrasings:
        _, dataset_pp = pp.Paraphraser(args).generate_paraphrasings(dataset, data_manager.get_df_pp())
        
    if args.translate:
        translate(dataset, dataset_pp, args)
        
    
    if args.test_knowledge_injection:
        ki_eval = ki.KnowledgeInjectionEval(args)
        ki_eval.run_eval(dataset, args.knowledge_inj_task)


if __name__ == '__main__':
    logging.info("Starting main")
    main()
