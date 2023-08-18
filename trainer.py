import os
import os.path
import pandas as pd
import random
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import get_overlap_thresholds, group_by_treatment, test_mask, Classifier, get_hidden_representations
from utils import collect_output_components , report_gpu, trace_counterfactual
from utils import geting_counterfactual_paths, get_single_representation, geting_NIE_paths
from data import test_restore_weight
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)
from data import ExperimentDataset, Dev, get_condition_inferences, get_inference_based, print_config, trace_optimized_params
from data import rank_losses, initial_partition_params, restore_original_weight, partition_param_train
from intervention import intervene, high_level_intervention
from analze import cma_analysis, compute_embedding_set, get_distribution, get_top_k
from utils import debias_test, get_nie_set_path
import yaml
from utils import get_num_neurons, get_params, get_diagnosis
from data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from data import exclude_grad
from transformers import Trainer
# from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, AdamW
from datasets import Dataset
import evaluate
import allennlp
from typing import Optional
from slanted_triangular import SlantedTriangular
from torch.optim import Adam


class CustomTrainer(Trainer):
    # Todo: custom where scheduler being created
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        cut_frac = 0.06

        # bugs: lr_scheuduler is no get_lr()[0]
        # Scheduler -> slanted override 
        # Allennlp -> override from
        if self.lr_scheduler is None:
            self.lr_scheduler = SlantedTriangular(optimizer = self.optimizer, 
                                           num_epochs = self.args.num_train_epochs,
                                           cut_frac= cut_frac,
                                           )
            self._created_lr_scheduler = True
        
        return self.lr_scheduler

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True) 

def preprocss(df):
    if '-' in df.gold_label.unique(): 
        df = df[df.gold_label != '-'].reset_index(drop=True)

    return df

def to_label_id(text_label): return label_maps[text_label]

def get_dataset(config, data_name = 'train_data'):
    df = pd.read_json(os.path.join(config['data_path'], config[data_name]), lines=True)
    df = preprocss(df)
    df_new = df[['sentence1', 'sentence2', 'gold_label']]
    df_new.rename(columns = {'gold_label':'label'}, inplace = True)
    df_new['label'] = df_new['label'].apply(lambda label_text: to_label_id(label_text))
    return Dataset.from_pandas(df_new)

def compute_metrics(eval_pred):
    # why compute_metrics never be called?
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    global tokenizer
    global label_maps
    global metric

    metric = evaluate.load("accuracy")

    with open("baseline_config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    dataset = {}
    tokenized_datasets = {}
    output_dir = '../models/baseline/'
    label_maps = {"entailment": 0, "contradiction": 1, "neutral": 2}
    
    
    tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
    model = BertForSequenceClassification.from_pretrained(config['model']["model_name"], num_labels = len(label_maps.keys()))


    for data_name in ["train_data", "validation_data", "test_data"]:
        dataset[data_name] = get_dataset(config, data_name = data_name)
        tokenized_datasets[data_name] = dataset[data_name].map(tokenize_function, batched=True)
        if data_name != 'test_data': tokenized_datasets[data_name].shuffle(seed=42)

    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    training_args = TrainingArguments(output_dir = output_dir,
                                      report_to="none",
                                      overwrite_output_dir = True,
                                      learning_rate = float(config['optimizer']['lr']),
                                      weight_decay = config['optimizer']['weight_decay'],
                                      per_device_train_batch_size = 32,
                                      num_train_epochs = config["num_epochs"],
                                      half_precision_backend = 'amp')
    
    # TODO: fix: change lr respect slanted triagular?
    # TODO: find where model is not able to learn?
    # TODO: fix optmizers
    opitmizer = AdamW(params=model.parameters(),
                      lr= float(config['optimizer']['lr']) , 
                      weight_decay = config['optimizer']['weight_decay'])

    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset= tokenized_datasets["train_data"],
        eval_dataset= tokenized_datasets["validation_data"],
        tokenizer =tokenizer, 
        compute_metrics=compute_metrics,
        optimizers = (opitmizer, None),
        )
    
    trainer.train()
    # allennlp train -> trainer( trainer.train() ) -> using scheduler
    # allennlp train ->  TrainModel(Registrable)
    # learning_rate_scheduler -> where a learning scheduler is used  or called ?
    # train_loop from_something is trainer inside this?; train_loop.run()  

if __name__ == "__main__":
    main()
     