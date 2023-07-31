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
    breakpoint()

    # Todo: combine to list
    # pair_sentences = [[premise, hypo] for premise, hypo in zip(cur_inputs['sentence1'], cur_inputs['sentence2'])]
    # pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")

    return True #tokenizer(examples["text"], padding="max_length", truncation=True)


# class TrainingDataset(Dataset):
#     def __init__(self, config, encode, DEBUG=False) -> None: 
#         df = pd.read_json(os.path.join(config['data_path'], config['train_data']), lines=True)
#         dataset = df[['sentence1', 'sentence2', 'gold_label']].to_dict()
#         # self.dataset = [{'label': row['gold_label'], 'text': [ row['sentence1'], row['sentence2']] } for index, row in df.iterrows()] 

def main():
    with open("train_config.yaml", "r") as yamlfile:
        # baseline config
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    tokenizer = AutoTokenizer.from_pretrained(config['dataset_reader']['tokenizer']['model_name'])
    model = BertForSequenceClassification.from_pretrained(config['model']['tokens']["model_name"], num_labels=3 )
    df = pd.read_json(os.path.join(config['data_path'], config['train_data']), lines=True)
    df_new = df[['sentence1', 'sentence2', 'gold_label']]
    train_set = Dataset.from_pandas(df_new)

    breakpoint()

    # train_dataset = TrainingDataset(config=config, encode=tokenizer)

    # with open("config.yaml", "r") as yamlfile:
    #     config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    #     print(config)
    
    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    
    # tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    # model = BertForSequenceClassification.from_pretrained(config["model_name"])
    
    breakpoint()
    
    # experiment_set = ExperimentDataset(config, encode = tokenizer)                            
    # dataloader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)
    
    # trainer = Trainer(
    #     model,
    #     training_args,
    #     train_dataset= dataloader,
    #     tokenizer=tokenizer,)


    # trainer.train()
    
    # trainer = Trainer(
    #     model,
    #     training_args,
    #     train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["validation"],
    #     data_collator=data_collator,
    #     tokenizer=tokenizer,)
        


if __name__ == "__main__":
    main()
     