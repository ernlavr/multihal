# import source code
import sys
sys.path.append("..")


import os
import sklearn
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import time

print(f"Starting analysis at {time.ctime()}")

def get_dataframes(result_dir):
    dataframes = []
    for dirpath, dirnames, filenames in os.walk(result_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            if "full" in filename:
                continue
            
            # process
            print(f"Reading {dirpath}/{filename}")
            
            df = pl.read_json(os.path.join(dirpath, filename), infer_schema_length=1000000)
            
            # if translations in columns
            if "translation" in df.columns:
                # remove the translation column
                df = df.drop("translation")
            
            # get the name of the file without the extension
            name = os.path.splitext(filename)[0]
            name = name.replace("llm_eval_", "")
            name = "_".join(name.split("_")[:-1])
            
            lang = "eng"
            if "deu" in name:
                lang = "deu"
            elif "fra" in name:
                lang = "fra"
            elif "spa" in name:
                lang = "spa"
            elif "ita" in name:
                lang = "ita"
            elif "por" in name:
                lang = "por"
            
            if lang in name:
                name = name.replace(f"_{lang}", "")
            
            task = "G-RAG" if "grag" in filename else "QA"
            # add the dataframe to a dictionary
            dataframes.append((name, lang, task, df))
    return dataframes

# dir of this file
this_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(this_dir, "main_results")
dfs = get_dataframes(result_dir)
print(f"Found {len(dfs)} dataframes")

# sort dfs first by languages [eng, de, es, fr, it, pt] and then by name
dfs = sorted(dfs, key=lambda x: (["eng", "deu", "fra", "ita", "spa", "por"].index(x[1]), x[0], x[2]))
[print(f"{i}: {df[0]} ({df[1]}, {df[2]})") for i, df in enumerate(dfs)]


###
### -------- Perform Vectara Hallucination Test --------
###
print(f"vectaring... {time.ctime()}")

from transformers import pipeline, AutoTokenizer
from collections import Counter

def vectara_test(classifier, df, grag=False):
    questions = df['input'].to_list()
    model_outputs = df["model_response"].to_list()
    ground_truths = df["output"].to_list()
    kg_paths = None
    zipped_data = list(zip(questions, model_outputs, ground_truths))
    
    # Prompt the pairs
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    input_pairs = [prompt.format(text1=f"{q}; {g_t}", text2=model_ans) 
                   for q, g_t, model_ans in zipped_data]
    
    if grag is True:
        kg_paths = df["trip_labels"].to_list()
        zipped_data = list(zip(questions, model_outputs, ground_truths, kg_paths))
        input_pairs = [prompt.format(text1=f"{kg_path}; {q}; {g_t}", text2=model_ans) 
                   for q, g_t, model_ans, kg_path in zipped_data]

    full_scores = classifier(input_pairs, top_k=1) # List[List[Dict[str, float]]]
    hallc_labels = [i[0]['label'] for i in full_scores]
    confidence_scores = [i[0]['score'] for i in full_scores]
    
    return hallc_labels, confidence_scores

    
# Use text-classification pipeline to predict
pipelne_model = pipeline(
        "text-classification",
        model='vectara/hallucination_evaluation_model',
        tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
        trust_remote_code=True
    )

results = []

for i in range(0, len(dfs), 1):
    if 'eng' not in dfs[i][1]:
        continue
    # print out condition, hallc labels % and confidence scores
    print(f"Condition: {dfs[i][0]} ({dfs[i][1]}-{dfs[i][2]})")
        
    # perform vectara test
    data: pl.DataFrame = dfs[i][3]
    hallc_labels, confidence_scores = vectara_test(pipelne_model, data, grag=True if 'rag' in dfs[i][2].lower() else False)
    
    data = data.with_columns(
        pl.Series("vectara_labels", hallc_labels),
        pl.Series("vectara_confidence", confidence_scores)
    )
    
    # create directory supplementary_experiments if it does not exist
    os.makedirs(os.path.join(this_dir, "supplementary_exp"), exist_ok=True)
    data.write_json(os.path.join(this_dir, "supplementary_exp", f"vectara_{dfs[i][0]}_{dfs[i][1]}_{dfs[i][2]}.json"))
    
        
    
###
### -------- Perform NLI Test
###
print(f"nling... {time.ctime()}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def get_entailment_percentage(model_name, dataset, device, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    number_of_entails = 0
    number_of_neutrals = 0
    number_of_contradictions = 0
    total_number_of_datapoints = len(dataset)

    label_names = ["entailment", "neutral", "contradiction"]

    premises = []
    hypotheses = []

    # Collect all examples first
    for i in dataset.iter_rows(named=True):
        premise = i['input'] + " " + i['output']
        hypothesis = i['model_response']
        premises.append(premise)
        hypotheses.append(hypothesis)

    # Process in batches
    labels = []
    for start in tqdm(range(0, total_number_of_datapoints, batch_size)):
        end = start + batch_size
        batch_premises = premises[start:end]
        batch_hypotheses = hypotheses[start:end]

        inputs = tokenizer(
            batch_premises,
            batch_hypotheses,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)  # [batch, 3]
            pred_labels = predictions.argmax(dim=-1).tolist()


        # convert IDs to labels
        labels.extend(pred_labels)
        

        # Count results
        for label_id in pred_labels:
            label = label_names[label_id]
            if label == "entailment":
                number_of_entails += 1
            elif label == "neutral":
                number_of_neutrals += 1
            elif label == "contradiction":
                number_of_contradictions += 1

    # add labels to dataset
    dataset = dataset.with_columns(
        pl.Series("nli_labels", [label_names[l] for l in labels])
    )

    percentage_entails = (number_of_entails / total_number_of_datapoints) * 100
    percentage_neutrals = (number_of_neutrals / total_number_of_datapoints) * 100
    percentage_contradictions = (number_of_contradictions / total_number_of_datapoints) * 100
    return dataset, percentage_entails, percentage_neutrals, percentage_contradictions
    

for i in range(0, len(dfs), 2):        
    # perform vectara test
    kg_rag = dfs[i][3]
    qa = dfs[i+1][3]
    
    # report percentage of entailments
    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    print(f"Using model {model_name} for entailment classification")
    ds_kgrag, kg_entails, kg_neutral, kg_contradiction = get_entailment_percentage(model_name, kg_rag, device)
    print(f"Percentage of entailments KG-RAG: {kg_entails:.2f}%; Neutral: {kg_neutral:.2f}%; Contradiction: {kg_contradiction:.2f}%")
    

    ds_qa, qa_entails, qa_neutral, qa_contradiction = get_entailment_percentage(model_name, qa, device)
    print(f"Percentage of entailments QA: {qa_entails:.2f}%; Neutral: {qa_neutral:.2f}%; Contradiction: {qa_contradiction:.2f}%")
    
    # save the datasets to json
    os.makedirs(os.path.join(this_dir, "supplementary_exp"), exist_ok=True)
    ds_kgrag.write_json(os.path.join(this_dir, "supplementary_exp", f"nli_{dfs[i][0]}_{dfs[i][1]}_{dfs[i][2]}.json"))
    ds_qa.write_json(os.path.join(this_dir, "supplementary_exp", f"nli_{dfs[i+1][0]}_{dfs[i+1][1]}_{dfs[i+1][2]}.json"))

# get scores of full dataset
print(f"Done at {time.ctime()}")