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
from transformers import TrainingArguments, Trainer
from datasets import Dataset

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

def main():
    global tokenizer
    global label_maps
    
    with open("train_config.yaml", "r") as yamlfile:
        # baseline config
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    dataset = {}
    tokenized_datasets = {}
    output_dir = '../models/baseline/'
    label_maps = {"entailment": 0, "contradiction": 1, "neutral": 2}
    tokenizer = AutoTokenizer.from_pretrained(config['dataset_reader']['tokenizer']['model_name'])
    model = BertForSequenceClassification.from_pretrained(config['model']['tokens']["model_name"], num_labels=3 )

    for data_name in ["train_data", "validation_data", "test_data"]:
        dataset[data_name] = get_dataset(config, data_name = data_name)
        tokenized_datasets[data_name] = dataset[data_name].map(tokenize_function, batched=True)
        if data_name != 'test_data': tokenized_datasets[data_name].shuffle(seed=42)

    # "trainer": {
    #   "num_epochs": 3,
    #   "validation_metric": "+accuracy",
    #   "learning_rate_scheduler": {
    #     "type": "slanted_triangular",
    #     "cut_frac": 0.06
    #   },
    #   "optimizer": {
    #     "type": "huggingface_adamw",
    #     "lr": 5e-5,
    #     "weight_decay": 0.1,
    #   },
    #   "use_amp": true,
    #   "cuda_device" : 0,
    # }
    
    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    training_args = TrainingArguments(output_dir = output_dir,
                                      overwrite_output_dir = True,
                                      learning_rate = 5e-5,
                                      weight_decay = 0.1,
                                      per_device_train_batch_size = 32,
                                      num_train_epochs = 3,
                                      half_precision_backend = 'amp')
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset= tokenized_datasets["train_data"],
        eval_dataset= tokenized_datasets["validation_data"],
        tokenizer =tokenizer,)
        
    breakpoint()


if __name__ == "__main__":
    main()
     