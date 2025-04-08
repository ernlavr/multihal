import json
from itertools import chain
import torch
import logging
from dateutil import parser
from datetime import datetime
import src.utils.helpers as hlp
import src.utils.constants as const
import string


def parse_flexible_date(date_str):
    try:
        date = parser.parse(date_str, fuzzy=False, default=datetime(1900, 1, 1))
        # if year is larger than 2100, return None
        if date.year > 2100:
            return None
        return date.strftime("%Y-%m-%d")  # Format output as "17 September 2009"
    except Exception as e:
        return None

def get_answer_types(answers: list):
    output = []
    for idx, i in enumerate(answers):
        # parse date
        to_parse = i.translate(str.maketrans('', '', string.punctuation.replace('.', '')))
        if (date_parsed := hlp.parse_flexible_date(to_parse)) is not None and len(to_parse) >= 4 and len(to_parse) <= 20:
            output.append(const.ANS_TYPE_DATE)
            continue
        
        number_parse = None
        try: number_parse = float(to_parse)
        except: pass
        
        if number_parse is not None:
            output.append(const.ANS_TYPE_NUMBER)
            continue        
        
        # add other
        else:
            output.append(const.ANS_TYPE_OTHER)
            
    return output

def remap_answer_types(answer: str):
    output = []
    mappings = {
        "rank": const.ANS_TYPE_RANK,
        "numeric": const.ANS_TYPE_NUMBER,
        "number": const.ANS_TYPE_NUMBER,
        "date": const.ANS_TYPE_DATE,
    }
    return mappings.get(answer, const.ANS_TYPE_OTHER)
        

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
    
def get_refusal_strings():
    """ Defines a list of strings that indicate a refusal to answer from an LLM.
    Normally happens for subjective nature questions.
    """
    return ["I'm an AI", "I have no comment", "As an AI language model", "I am an", "I do not have", "I don't have", "I am an artificial intelligence", "Nothing happens", "nothing in particular"]
    
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
    if torch.cuda.is_available():
        logging.info(f"Cuda available: {torch.cuda.is_available()}; \
                    Device Count: {torch.cuda.device_count()}; \
                    Current Device {torch.cuda.current_device()}; \
                    Device Name {torch.cuda.get_device_name(0)}")
    