import os
import datasets
from tqdm import tqdm
import pickle
import hashlib
from src.utils.decorators import cache_decorator

@cache_decorator("loaded_datasets")
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
    }

    # Determine which datasets to load
    datasets_to_load = args.datasets if args.datasets else dataset_loaders.keys()
    loaded_datasets = {name: loader() for name, loader in tqdm(dataset_loaders.items(), "Loading datasets") if name in datasets_to_load}
    return loaded_datasets

    

def _load_shroom2024(task='all'):
    """ Loads the shroom2024 dataset """
    path = 'res/shroom2024/SHROOM_unlabeled-training-data-v2/train.model-aware.v2.json'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found. Please download the datasets with ./getData.sh")
    data = datasets.load_dataset('json', data_files=path)

    if task != 'all':
        data = data.filter(lambda x: x['task'].lower() == task)

    return data

def _load_shroom2025():
    """ Loads the shroom2025 dataset """
    path = 'res/shroom2025/train/mushroom.en-train_nolabel.v1.jsonl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found. Please download the datasets with ./getData.sh")
    data = datasets.load_dataset('json', data_files=path)
    data.add_column('task', ['all'] * data.shape[0])
    return data

def _load_halueval(task='qa'):
    """ Loads the halueval dataset """
    data = datasets.load_dataset('pminervini/HaluEval', task)
    data.add_column('task', [task] * data.shape[0])
    data['val'] = data.pop('data') # for consistency
    return data

def _load_truthfulqa_gen():
    """ Loads the truthfulqa dataset generative split """
    data = datasets.load_dataset('truthfulqa/truthful_qa', 'generation')
    data.add_column('task', ['qa'] * data.shape[0])
    data['val'] = data.pop('validation') # for consistency
    return data

def _load_felm():
    """ Loads the FELM dataset """
    domain = 'wk'
    data = datasets.load_dataset('hkust-nlp/felm', domain)
    data['val'] = data.pop('test') # for consistency
    data.add_column('task', ['qa'] * data.shape[0])
    return data

def _load_halubenchmark():
    """ Loads the HaluBenchmark dataset """
    data = datasets.load_dataset('PatronusAI/HaluBench')
    data['val'] = data.pop('test') # for consistency
    data.add_column('task', ['qa'] * data.shape[0])
    return data

def _load_defan():
    """ Loads the DefAN dataset """
    path = 'res/defan'
    data_files = sorted([os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')])
    output = datasets.DatasetDict({'val': None})
    for file in data_files:
        data = datasets.load_dataset('csv', data_files=file)
        domain = file.split('_')[2]
        data['train'] = data['train'].add_column('domain', [domain] * data['train'].shape[0])
        data['val'] = data.pop('train')
        if output['val'] is None:
            output['val'] = data['val']
        else:
            output['val'] = datasets.concatenate_datasets([output.get('val'), data['val']])

    data.add_column('task', ['qa'] * data.shape[0])
    # data = datasets.load_dataset('csv', data_files=data_files)
    # data['val'] = data.pop('train') # for consistency
    return output

def _load_simpleQa():
    """ Loads the SimpleQA dataset """
    data = datasets.load_dataset('csv', data_files='res/simpleqa/simple_qa_test_set.csv')
    data.add_column('task', ['qa'] * data.shape[0])
    data['val'] = data.pop('train') # for consistency
    return data