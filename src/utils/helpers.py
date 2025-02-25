import json
from itertools import chain
import torch
import logging

def remove_starting_pronouns(text: str):
    pronouns = ['the']
    for p in pronouns:
        if text.lower().startswith(p):
            text = text[len(p):]
    return text.strip()

def fill_ignore_properties(path):
    with open(path, "r") as f:
        data = json.load(f)
        data = [d['id'] for d in data]
        return set(data)
    
def fill_all_properties(path):
    with open(path, "r") as f:
        data = json.load(f)
        
        # check for duplicate ids data['id']
        ids = [item['id'] for item in data]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate Ids")

        data = {item['id']: item for item in data}
        return data

    
def flatten_if_2d(lst):
    return list(chain.from_iterable(lst)) if any(isinstance(i, list) for i in lst) else lst

def print_cuda_stats():
    logging.info(f"Cuda available: {torch.cuda.is_available()}; \
                Device Count: {torch.cuda.device_count()}; \
                Current Device {torch.cuda.current_device()}; \
                Device Name {torch.cuda.get_device_name(0)}")
    