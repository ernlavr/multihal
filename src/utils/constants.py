import src.utils.helpers as uti
import json

def get_kg_mappings():
    mappings = {
        'qsranking': ['wikidata'], 
        'healthcare': ['pubmed'], 
        'science and technology': ['wikidata'], 
        'entertainment': [], 
        'ragtruth': [], 
        'politics': [],
        'nobleprize': [], 
        'art': [], 
        'census': [], 
        'finance': [], 
        'general': [], 
        'other': [], 
        'sports': [], 
        'geography': []
        }
    return mappings

ANS_TYPE_OTHER = "other"
ANS_TYPE_DATE = "date"
ANS_TYPE_NUMBER = "number"
PROPS_TIMED_PATH = "res/wd_properties_to_ignore/time_based_props.json"
PROPS_NUM_PATH = "res/wd_properties_to_ignore/numerical_props.json"

def get_list_of_props_to_ignore():
    return uti.fill_ignore_properties("res/wd_properties_to_ignore/ids_to_remove_V2.json")

def get_list_of_timed_props(path):
    output = {}
    with open(path, "r") as f:
        data = json.load(f)
        for p in data['data']:
            prop = p['property'].split('/')[-1]
            label = p['propertyLabel']
            if "DEPRECATED" not in label:
                output[prop] = label
    return output