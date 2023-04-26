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
from data import ExperimentDataset, Dev, get_predictions, print_config
from intervention import intervene, high_level_intervention
from analze import cma_analysis, compute_embedding_set, get_distribution, get_top_k
from utils import debias_test
import yaml

def main():

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(config)

    
    DEBUG = True
    debug = False # for tracing top counterfactual 
    torch.manual_seed(config['seed'])
    collect_representation = True
    mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
    save_nie_set_path = f'../pickles/class_level_nie_{config["num_samples"]}_samples.pickle' if config['is_group_by_class'] else f'../pickles/nie_{config["num_samples"]}_samples.pickle'
    
    if   config["dev-name"] == 'mismatched': config["dev_json"]['mismatched'] = 'multinli_1.0_dev_mismatched.jsonl'
    elif config["dev-name"] == 'hans':       config["dev_json"]['hans'] = 'heuristics_evaluation_set.jsonl' 
    elif config["dev-name"] == 'matched':    config["dev_json"]['matched'] = 'multinli_1.0_dev_matched.jsonl'

    geting_counterfactual_paths(config)
    geting_NIE_paths(config)
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"])
    model = model.to(DEVICE)
        
    # Todo: generalize for every model 
    
    # using same seed everytime we create HOL and LOL sets 
    experiment_set = ExperimentDataset()                            
    dataloader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)

    if config['getting_counterfactual']: collect_output_components(config, DEVICE = DEVICE)
    
    print_config(config)

    if not os.path.isfile(save_nie_set_path):

        combine_types = []
        pairs = {}

        if config['is_group_by_class']:
            
            nie_dataset = {}
            nie_loader = {}

            for type in ["contradiction","entailment","neutral"]:
            
                # get the whole set of validation 
                pairs[type] = list(experiment_set.df[experiment_set.df.gold_label == type].pair_label)
                
                # samples data
                ids = list(torch.randint(0, len(pairs[type]), size=(config['num_samples'] //3,)))
                pairs[type] = np.array(pairs[type])[ids,:].tolist()
                
                nie_dataset[type] = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(pairs[type])]
                nie_loader[type] = DataLoader(nie_dataset[type], batch_size=32)

        else:

            # balacing nie set across classes
            for type in ["contradiction","entailment","neutral"]:
            
                # get the whole set of validation 
                pairs[type] = list(experiment_set.df[experiment_set.df.gold_label == type].pair_label)
                
                # samples data
                ids = list(torch.randint(0, len(pairs[type]), size=(config['num_samples'] //3,)))
                
                pairs[type] = np.array(pairs[type])[ids,:].tolist()
                combine_types.extend(pairs[type])

            # dataset = [([premise, hypo], label) for idx, (premise, hypo, label) in enumerate(pairs['entailment'])]
            nie_dataset = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(combine_types)]
            nie_loader = DataLoader(nie_dataset, batch_size=32)
        
        with open(save_nie_set_path, 'wb') as handle:
            pickle.dump(nie_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(nie_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Done saving NIE set  into {save_nie_set_path} !")

    if config['analysis']:  cma_analysis(save_nie_set_path = save_nie_set_path, model = model, treatments = mode, tokenizer = tokenizer, experiment_set = experiment_set, DEVICE = DEVICE, DEBUG = True)
    if config['topk']: return None if sum(config['is_NIE_exist']) != len(config['is_NIE_exist']) else get_top_k(config, treatments=mode) 
    if config['embedding_summary']: compute_embedding_set(experiment_set, model, tokenizer, DEVICE)
    if config['distribution']: get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)
    if config['debias']: debias_test(config, model, experiment_set, tokenizer, DEVICE)
    if config['traced']: trace_counterfactual(model, save_nie_set_path, tokenizer, DEVICE, debug)
    if config['is_prediction']: get_predictions(model, tokenizer, DEVICE)

if __name__ == "__main__":
    main()

