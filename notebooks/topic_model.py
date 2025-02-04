import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import sklearn.metrics
import transformers
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from datetime import datetime
import wandb


class DataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        print(f"Shape of the dataframe: {self.df.shape}; Datasets: {self.df['source_dataset'].unique()}")

    def clean_data(self):
        df = self.df.sort_values(by='context')
        df = df.dropna(subset=['context'])
        df['context_length'] = df['context'].apply(len)
        df = df.sort_values(by='context_length')
        contexts_to_remove = ['false stereotype', '91 = 7 * 13', 'subjective', 'tautology', 'indexical']
        df = df[~df['context'].isin(contexts_to_remove)]
        print(f"Remaining samples: {len(df)}")
        return df


class DatasetHandler:
    def __init__(self, df, domains):
        self.df = df
        self.domains = [d.strip() for d in domains]
        self.label_mapping = {}
        self.int_to_label = {}

    def prepare_datasets(self):
        training_data = self.df[self.df['domain'].isin(self.domains)]
        test_data = self.df[~self.df['domain'].isin(self.domains)]
        labels = training_data['domain'].values
        self.label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        self.int_to_label = {idx: label for label, idx in self.label_mapping.items()}
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            training_data['input'].values, [self.label_mapping[label] for label in labels], test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, test_data

    def compute_class_weights(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device) if self.class_weights is not None else None)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class ModelTrainer:
    def __init__(self, model_name, label_to_int, int_to_label):
        self.model_name = model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        self.model = None

    def preprocess_data(self, texts, labels):
        dataset = Dataset.from_dict({'text': texts, 'label': labels})
        return dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, padding=True, max_length=512, return_tensors='pt'), batched=True)
    
    def inference(self, df):
        from torch.utils.data import DataLoader

        inputs = list(df['input'].values)
        labels = df['domain'].values

        preprocessed_data = self.preprocess_data(inputs, labels)

        # Creating a DataLoader for batching
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(preprocessed_data['input_ids']), 
            torch.tensor(preprocessed_data['attention_mask'])
        )
        dataloader = DataLoader(dataset, batch_size=8)

        logits = []

        # Iterating over the DataLoader for batched inference
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradients during inference
            for batch in dataloader:
                input_ids, attention_mask = batch

                # put input ids to same device as model
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)

                # Perform inference
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits.append(outputs.logits)

        # Concatenate logits from all batches
        logits = torch.cat(logits, dim=0)
        logits = logits.detach().cpu().numpy()

        # Convert logits to predictions
        predictions = np.argmax(logits, axis=1)

        # add predictions and labels to the dataframe
        df['_predictions'] = [self.int_to_label[pred] for pred in predictions]
        df['_labels'] = labels

        # log df to wandb
        wandb.log({"Test Data Predictions": wandb.Table(dataframe=df)})
        return df

    def train_model(self, X_train, X_test, y_train, y_test, class_weights):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.label_to_int))
        training_args = TrainingArguments(
            eval_strategy="steps", 
            eval_steps=100,
            learning_rate=1e-5, 
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8, 
            num_train_epochs=5, 
            weight_decay=0.01, 
            metric_for_best_model="f1",
            load_best_model_at_end=True,
            
            output_dir="./models", 
            report_to="wandb"
        )
        trainer = WeightedTrainer(
            class_weights=class_weights, model=model, args=training_args,
            train_dataset=self.preprocess_data(X_train, y_train),
            eval_dataset=self.preprocess_data(X_test, y_test),
            data_collator=transformers.DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        self.model = model


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Compute confusion matrix
        cm = sklearn.metrics.confusion_matrix(labels, predictions)

        cn = [self.int_to_label[i] for i in list(self.int_to_label.keys())]
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=labels, preds=predictions,
                        class_names=cn)})

        return {
            'eval_f1': sklearn.metrics.f1_score(labels, predictions, average='weighted'),
            'eval_accuracy': sklearn.metrics.accuracy_score(labels, predictions),
            'eval_precision': sklearn.metrics.precision_score(labels, predictions, average='weighted'),
            'eval_recall': sklearn.metrics.recall_score(labels, predictions, average='weighted')
        }


def train():
    """ Function to train the model with different hyperparameters """
    wandb.init()  # Initialize a W&B run
    config = wandb.config  # Access sweep hyperparameters

    # Load and preprocess dataset
    data_processor = DataProcessor("/Users/dr84sy/Documents/PhD/projects/multihal/multihal/output/data/multihal_unprocessed.csv")
    cleaned_data = data_processor.clean_data()

    domains = ['qsranking', 'healthcare', 'science and technology', 'entertainment', 'ragtruth', 'politics',
               'nobleprize', 'art', 'census', 'finance', 'general', 'other', 'sports', 'geography']
    
    dataset_handler = DatasetHandler(cleaned_data, domains)
    X_train, X_val, y_train, y_val, X_test = dataset_handler.prepare_datasets()
    class_weights = dataset_handler.compute_class_weights(y_train)

    # Train model with selected parameters
    model_trainer = ModelTrainer(config.model_name, dataset_handler.label_mapping, dataset_handler.int_to_label)
    model_trainer.train_model(
        X_train, X_val, y_train, y_val, class_weights
    )

    model_trainer.inference(X_test)
    wandb.finish()


# Initialize W&B sweep configuration
sweep_config = {
    "method": "grid",  # Can be "random" or "bayes" for more optimization
    "metric": {"name": "eval/f1", "goal": "maximize"},
    "parameters": {
        "model_name": {
            "values": ["bert-base-uncased", "bert-large-uncased", "FacebookAI/roberta-large"]
        }
    }
}

# Create a sweep
sweep_id = wandb.sweep(sweep_config, project="transformer-sweep")
# Run the sweep
wandb.agent(sweep_id, train, count=10)  # Runs 10 experiments

