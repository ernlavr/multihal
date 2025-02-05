
import sklearn
import datasets

import transformers
from transformers import DataCollatorWithPadding
import numpy as np
import torch
import torch.nn as nn
import wandb
import os
import random

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

# class CustomTrainer(transformers.Trainer):
#     def __init__(self, class_weights=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # create tensor out of class_weight values
#         self.class_weights = torch.tensor([i for i in class_weights.values()], dtype=torch.float32).to(self.args.device)
        

    
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # compute custom loss (suppose one has 3 labels with different weights)
#         loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {DEVICE}")

def enforce_reproducibility(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

enforce_reproducibility(42)

def prepare_deep_learning(model_name: str, 
                          weights,
                          train_corpus: np.ndarray, 
                          test_corpus: np.ndarray, 
                          train_labels: np.ndarray, 
                          test_labels: np.ndarray):

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, predictions, average='weighted')
        return { 'eval/f1': f1, 'eval/accuracy': accuracy, 'eval/precision': precision, 'eval/recall': recall }


    # tokenize
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    train_ds = datasets.Dataset.from_dict({'text': train_corpus, 'label': train_labels})
    test_ds = datasets.Dataset.from_dict({'text': test_corpus, 'label': test_labels})

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    # encode
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    # train
    epochs = 4
    batch_size = 8
    lr = 0.0001
    training_args = transformers.TrainingArguments(
                        eval_strategy = "steps",
                        eval_steps=100,
                        learning_rate=lr,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        num_train_epochs=epochs,
                        weight_decay=0.01,
                        metric_for_best_model="f1",
                        save_strategy="no",
                        output_dir="./models",
                    )

    trainer = transformers.Trainer(
        # class_weights=weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    # model training on device:
    print(f"Training model {model_name} on {model.device}")

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
        return lambda x, y, q, z: prepare_deep_learning('google-bert/bert-base-uncased', cw, x, y, q, z)
    elif name == "roberta":
        return lambda x, y, q, z: prepare_deep_learning('FacebookAI/xlm-roberta-large', cw, x, y, q, z)




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
os.environ['WANDB_API_KEY'] = '49c8e7dca82f91f9d65021c3dd71101b686c1f53'


sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'eval/f1',
        'goal': 'maximize'
    },
    'parameters': {
        'model': {
            'values': ['svm', 'roberta', 'bert', 'logistic', 'nb', 'rf']
        },
        'vectorizer': {
            'values': ['count', 'tfidf', 'minilm']
        },
        'question_w_cntx': {
            'values': [True, False]
        },
        'val_split': {
            'values': [0.05, 0.1, 0.2, 0.3]
        }
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
            corpus =  (dataframe['question'] + ' ' + dataframe['context']).values
        else:
            corpus =  (dataframe['question']).values

        # set corpus, for deep learning we need to keep the original corpus
        if 'bert' not in config.model:
            print("Corpus vectorized!")
            corpus = feature_vectorizer(corpus)
            
        train_corpus, test_corpus, train_labels, test_labels = train_test_split(corpus, labels, test_size=config.val_split, random_state=42)

        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
        class_weights = {0: class_weights[0], 1: class_weights[1]}

        model = get_classification_model(config.model, cw=class_weights)

        if 'bert' in config.model:
            trainer = model(train_corpus, test_corpus, train_labels, test_labels)
            trainer.train()
        
        else:
            model.fit(train_corpus, train_labels)
            predictions = model.predict(test_corpus)

            print(classification_report(test_labels, predictions, target_names=[decode_label(0), decode_label(1)]))
            precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(test_labels, predictions, average='weighted')

            run.log({'eval/accuracy': sklearn.metrics.accuracy_score(test_labels, predictions)})
            run.log({'eval/precision': precision})
            run.log({'eval/recall': recall})
            run.log({'eval/f1': f1})

wandb.agent(sweep_id, train)