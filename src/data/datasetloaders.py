import os
import datasets
from tqdm import tqdm
import pickle
import hashlib
from src.utils.decorators import cache_decorator
import src.data.column_mapper as cm
import src.utils.config as config
import json
import copy
import numpy as np
import time
import polars as pl

def load_data(args):
    """ Returns all possible datasets unless there are explicit datasets specified
        under args.datasets. 
    """
    dataset_loaders = {
        'shroom2024': _load_shroom2024,
        'shroom2025': _load_shroom2025,
        'halueval': _load_halueval,
        'tqa_gen': _load_truthfulqa_gen,
        'felm': _load_felm,
        'halubench': _load_halubenchmark,
        'defan': _load_defan,
        'simpleqa': _load_simpleQa,
        
        'drop': _loadDrop
    }

    # Determine which datasets to load
    if args.datasets == 'all':
        datasets_to_load = dataset_loaders.keys()
    else:
        datasets_to_load = args.datasets
    
    loaded_datasets = {name: loader() for name, loader in tqdm(dataset_loaders.items(), "Loading datasets") if name in datasets_to_load}
    return loaded_datasets

@cache_decorator("shroom2024")
def _load_shroom2024(task='all'):
    """ Loads the shroom2024 dataset """
    path = 'res/shroom2024/SHROOM_unlabeled-training-data-v2/train.model-aware.v2.json'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found. Please download the datasets with ./getData.sh")
    data = datasets.load_dataset('json', data_files=path)

    if task != 'all':
        data = data.filter(lambda x: x['task'].lower() == task)

    return data

@cache_decorator("shroom2025")
def _load_shroom2025():
    """ Loads the shroom2025 dataset """
    path = 'res/shroom2025/train/mushroom.en-train_nolabel.v1.jsonl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found. Please download the datasets with ./getData.sh")
    data = datasets.load_dataset('json', data_files=path)

    # Add extra columns
    data['train'] = data['train'].add_column('task', ['qa'] * data['train'].shape[0])
    return data

@cache_decorator("halueval")
def _load_halueval(task='qa'):
    """ Loads the halueval dataset """
    data = datasets.load_dataset('pminervini/HaluEval', task)
    data['val'] = data.pop('data') # for consistency

    # Add extra columns
    data['val'] = data['val'].add_column('task', [task] * data['val'].shape[0])

    # TODO: Possibly fix this with domain modeling, original paper they get knowledge from Wikipedita
    data['val'] = data['val'].add_column('domain', [task] * data['val'].shape[0])
    return data

@cache_decorator("truthfulqa_gen")
def _load_truthfulqa_gen():
    """ Loads the truthfulqa dataset generative split """
    data = datasets.load_dataset('truthfulqa/truthful_qa', 'generation')
    data['val'] = data.pop('validation') # for consistency
    
    # Add extra columns
    data['val'] = data['val'].add_column('task', ['qa'] * data['val'].shape[0])
    return data

@cache_decorator("felm")
def _load_felm():
    """ Loads the FELM dataset """
    domain = 'wk'
    data = datasets.load_dataset('hkust-nlp/felm', domain)
    data['val'] = data.pop('test') # for consistency

    # Add extra columns
    data['val'] = data['val'].add_column('domain', [domain] * data['val'].shape[0])
    return data

@cache_decorator("halubenchmark")
def _load_halubenchmark():
    """ Loads the HaluBenchmark dataset """
    data = datasets.load_dataset('PatronusAI/HaluBench')
    data['val'] = data.pop('test') # for consistency
    
    # Add extra columns
    data['val'] = data['val'].add_column('task', ['qa'] * data['val'].shape[0])
    return data

@cache_decorator("defan")
def _load_defan() -> datasets.DatasetDict:
    """Loads and processes the DefAN dataset.
        DefAn contains paraphrased versions of a prompt, 1 question, 14 paraphrases.
        Therefore, we also capture the paraphrasings seperately with reference to first question.
    """
    BATCH_SIZE, PARAPHRASE_SIZE = 15, 14
    DEF_AN_DIR = 'res/defan'
    output = datasets.DatasetDict({'val': None, 'val_paraphrased': None})
    
    for file in sorted(os.path.join(DEF_AN_DIR, f) for f in os.listdir(DEF_AN_DIR) if f.endswith('.csv')):
        # Load and process file
        data = datasets.load_dataset('csv', data_files=file)['train']
        domain = os.path.basename(file).split('_')[2]
        data = data.add_column('domain', [domain] * len(data)) \
                  .add_column('id', range(len(data))) \
                  .add_column('task', ['qa'] * len(data))
        
        # Create val and paraphrased dictionaries
        val = {k: [] for k in data.column_names}
        val_pp = {k: [] for k in data.column_names}
        
        # Fill dictionaries
        for idx in range(0, len(data), BATCH_SIZE):
            # Add validation sample
            for col in data.column_names:
                val[col].append(f"defan_{idx}" if col == 'id' else data[idx][col])
                val_pp[col].extend([f"defan_{idx}" if col == 'id' else data[i][col] 
                                  for i in range(idx + 1, idx + BATCH_SIZE)])
        
        # Convert to datasets
        current = {
            'val': datasets.Dataset.from_dict(val),
            'val_paraphrased': datasets.Dataset.from_dict(val_pp)
        }
        
        # Verify dataset integrity
        val_ids = current['val']['id']
        pp_ids = np.array(current['val_paraphrased']['id'])
        assert len(pp_ids) / PARAPHRASE_SIZE == len(val_ids), "Paraphrased dataset size mismatch"
        for i in range(len(val_ids)):
            assert np.all(val_ids[i] == pp_ids[i * PARAPHRASE_SIZE:(i + 1) * PARAPHRASE_SIZE]), \
                   f"ID mismatch at index {i}"
        
        # Combine datasets
        if output['val'] is None:
            output.update(current)
        else:
            for key in output:
                output[key] = datasets.concatenate_datasets([output[key], current[key]])
    return output
    

@cache_decorator("simpleqa")
def _load_simpleQa():
    """ Loads the SimpleQA dataset """
    data = datasets.load_dataset('csv', data_files='res/simpleqa/simple_qa_test_set.csv')
    data['val'] = data.pop('train') # for consistency

    # Add extra columns
    data['val'] = data['val'].add_column('task', ['qa'] * data['val'].shape[0])
    metadata = data['val']['metadata']
    for idx, i in enumerate(metadata):
        metadata[idx] = eval(i.lower())

    data['val'] = data['val'].add_column('domain', [i['topic'] for i in metadata])
    sep = config.LIST_SEP
    data['val'] = data['val'].add_column('context', [sep.join(i['urls']) for i in metadata])

    return data

@cache_decorator("drop")
def _loadDrop():
    """ Loads the DROP dataset """
    data = datasets.load_dataset('ucinlp/drop')
    return data