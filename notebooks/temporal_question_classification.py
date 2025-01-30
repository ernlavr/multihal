
import sklearn
import datasets

import transformers
from transformers import DataCollatorWithPadding
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.utils.validation import check_is_fitted

import sentence_transformers as st
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.neighbors


def prepare_deep_learning(model_name: str, 
                          train_corpus: np.ndarray, 
                          test_corpus: np.ndarray, 
                          train_labels: np.ndarray, 
                          test_labels: np.ndarray):

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, predictions, average='weighted')
        return { 'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall }


    # tokenize
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    train_ds = datasets.Dataset.from_dict({'text': train_corpus, 'label': train_labels})
    test_ds = datasets.Dataset.from_dict({'text': test_corpus, 'label': test_labels})

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    # encode
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # train
    epochs = 4
    batch_size = 8
    lr = 0.0001
    training_args = transformers.TrainingArguments(
                        eval_strategy = "steps",
                        learning_rate=lr,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        num_train_epochs=epochs,
                        weight_decay=0.01,
                        metric_for_best_model="f1",
                        output_dir="./models",
                    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return trainer


def get_labels():
    labels = {"temporal": 1, "static": 0}
    return {**labels, **{v: k for k, v in labels.items()}}  # Merge both mappings

def decode_label(value):
    return get_labels()[value]

def add_labels(data: datasets.Dataset, split: str, label) -> datasets.Dataset:
    return data[split].add_column('label', [decode_label(label)] * len(data[split]))

def get_feature_vectorizer(name: str, tokenizer=None):
    if name == "count":
        return CountVectorizer(tokenizer=tokenizer).fit_transform
    elif name == "tfidf":
        return TfidfVectorizer(tokenizer=tokenizer).fit_transform
    elif name == "stella":
        return lambda x: st.SentenceTransformer('NovaSearch/stella_en_400M_v5', trust_remote_code=True).encode(x, show_progress_bar=True)
    elif name == 'cde':
        return lambda x: st.SentenceTransformer('jxm/cde-small-v2', trust_remote_code=True).encode(x, show_progress_bar=True)
    elif name == 'minilm':
        return lambda x: st.SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', trust_remote_code=True).encode(x, show_progress_bar=True)
    
def get_classification_model(name: str, cw=None):
    if name == "svm":
        return sklearn.svm.SVC(class_weight=cw)
    elif name == "logistic":
        return sklearn.linear_model.LogisticRegression()
    elif name == "nb":
        return sklearn.naive_bayes.MultinomialNB()
    elif name == "rf":
        return sklearn.ensemble.RandomForestClassifier(class_weight=cw)
    elif name == "bert":
        return lambda x, y, q, z: prepare_deep_learning('google-bert/bert-base-uncased', x, y, q, z)




temporal_data = datasets.load_dataset('copenlu/dynamicqa', 'temporal')
static_data = datasets.load_dataset('copenlu/dynamicqa', 'static')
temporal_data['test'] = add_labels(temporal_data, 'test', 'temporal')
static_data['test'] = add_labels(static_data, 'test', 'static')
data = datasets.concatenate_datasets([temporal_data['test'], static_data['test']]).shuffle()
data = datasets.DatasetDict({'train': data})


feature_vectorizer = get_feature_vectorizer('count', tokenizer=None)

dataframe = data['train'].to_pandas()
labels = dataframe['label'].values
corpus =  (dataframe['question']).values
train_corpus, test_corpus, train_labels, test_labels = train_test_split(corpus, labels, test_size=0.2, random_state=42)
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(text):
    return tokenizer.tokenize(text)


# setup a weights and bias sweep
import wandb
import os
os.environ['WANDB_API_KEY'] = '49c8e7dca82f91f9d65021c3dd71101b686c1f53'


sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'f1',
        'goal': 'maximize'
    },
    'parameters': {
        'vectorizer': {
            'values': ['count'] #, 'tfidf', 'minilm']
        },
        'question_w_cntx': {
            'values': [True] #, False]
        },
        'val_split': {
            'values': [0.05, 0.1] # , 0.2, 0.3
        },
        'model': {
            'values': ['bert', 'svm'] # 'logistic', 'nb', 'rf', 
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project='temporal_questions_multihal')

def train():
    with wandb.init() as run:
        config = wandb.config
        
        feature_vectorizer = get_feature_vectorizer(config.vectorizer, tokenizer=tokenize)

        dataframe = data['train'].to_pandas()
        labels = dataframe['label'].values
        if config.question_w_cntx:
            corpus =  (dataframe['context'] + ' ' + dataframe['question']).values
        else:
            corpus =  (dataframe['question']).values

        # set corpus, for deep learning we need to keep the original corpus
        vectorized_corpus = feature_vectorizer(corpus)
        if config.model == 'bert':
            vectorized_corpus = corpus
        
        train_corpus, test_corpus, train_labels, test_labels = train_test_split(vectorized_corpus, labels, test_size=config.val_split, random_state=42)

        class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_labels)
        class_weights = {0: class_weights[0], 1: class_weights[1]}

        model = get_classification_model(config.model, cw=class_weights)

        if config.model == 'bert':
            trainer = model(train_corpus, test_corpus, train_labels, test_labels)
            trainer.train()
        
        else:
            model.fit(train_corpus, train_labels)
            predictions = model.predict(test_corpus)

            print(classification_report(test_labels, predictions, target_names=[decode_label(0), decode_label(1)]))

            run.log({'accuracy': sklearn.metrics.accuracy_score(test_labels, predictions)})
            run.log({'f1': sklearn.metrics.f1_score(test_labels, predictions)})
            run.log({'precision': sklearn.metrics.precision_score(test_labels, predictions)})

wandb.agent(sweep_id, train)