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
from utils import debias_test, get_nie_set_path
import yaml
from utils import get_num_neurons, get_params

def main():

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(config)
    
    DEBUG = True
    debug = False # for tracing top counterfactual 
    torch.manual_seed(config['seed'])
    collect_representation = True
    mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_nie_set_path = f'../pickles/class_level_nie_{config["num_samples"]}_samples.pickle' if config['is_group_by_class'] else f'../pickles/nie_{config["num_samples"]}_samples.pickle'
    
    if config["dev-name"] == 'mismatched': config["dev_json"]['mismatched'] = 'multinli_1.0_dev_mismatched.jsonl'
    elif config["dev-name"] == 'hans': config["dev_json"]['hans'] = 'heuristics_evaluation_set.jsonl' 
    elif config["dev-name"] == 'matched': config["dev_json"]['matched'] = 'multinli_1.0_dev_matched.jsonl'

    geting_counterfactual_paths(config)
    geting_NIE_paths(config,mode)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"])
    model = model.to(DEVICE)

    # Todo: generalize for every model 
    # using same seed everytime we create HOL and LOL sets 
    experiment_set = ExperimentDataset(config, encode = tokenizer)                            
    dataloader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)

    if config['getting_counterfactual']: collect_output_components(config, DEVICE = DEVICE)
    if config['print_config']: print_config(config)
    if not os.path.isfile(save_nie_set_path): get_nie_set_path(config, experiment_set, save_nie_set_path)
    if config['analysis']:  cma_analysis(config, save_nie_set_path = save_nie_set_path, model = model, treatments = mode, tokenizer = tokenizer, experiment_set = experiment_set, DEVICE = DEVICE, DEBUG = True)
    # if config['topk']: print(f"the NIE paths are not available !") if sum(config['is_NIE_exist']) != len(config['is_NIE_exist']) else get_top_k(config, treatments=mode) 
    if config['topk']: get_top_k(config, treatments=mode) 
    if config['embedding_summary']: compute_embedding_set(experiment_set, model, tokenizer, DEVICE)
    if config['distribution']: get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)
    if config['debias']: debias_test(config, model, experiment_set, tokenizer, DEVICE)
    if config['traced']: trace_counterfactual(model, save_nie_set_path, tokenizer, DEVICE, debug)
    if config['get_prediction']: get_predictions(config, mode[0], model, tokenizer, DEVICE)
    
    # Todo: get softmax score for each sample from two techniques intervention and no intervention:  
    # 5 outputs  for each calss on both mnli and hans
    
    print(f"{config['label_maps']}")
    
    SAMPLES = 5
    
    for dev in ['mismatched', 'hans']:
        
        # get raw  distributions of intervention
        value = config["masking_rate"]

        key = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
        layer = config['layer']
        do = 'High-overlap'

        params, digits = get_params(config)
        
        if layer == -1:
            raw_distribution_path = f'raw_distribution_{key}_{do}_all_layers_{value}-k_{config["intervention_type"]}_{dev}.pickle'  
        else:
            raw_distribution_path = f'raw_distribution_{key}_{do}_L{layer}_{value}-k_{config["intervention_type"]}_{dev}.pickle'

        prediction_path = '../pickles/prediction/' 
        epsilon_path = f"v{round(params['epsilons'][0], digits['epsilons'])}"
        
        raw_distribution_path = os.path.join(os.path.join(prediction_path, epsilon_path),  raw_distribution_path)

        # if dev == "hans": raw_distribution_path = '../pickles/prediction/v0.7/raw_distribution_0.7_High-overlap_all_layers_0.05-k_weaken_hans.pickle'
        
        with open(raw_distribution_path, 'rb') as handle: 
            
            distributions = pickle.load(handle)
            golden_answers = pickle.load(handle)



        # Todo: get index of current labels
        print(f" ++++++++  {dev} set, masking rate {value}, weaken rate : {key} ++++++++")
        # print(f"cur path : {raw_distribution_path}")
        
        for mode in ['Null', 'Intervene']:

            print(f">> {mode}")
        
            cur_labels = set(golden_answers[mode])
            
            dists = {}
           
            for idx, label in enumerate(golden_answers[mode]): 
                
                if label not in dists.keys(): dists[label] = []

                dists[label].append(distributions[mode][idx])
        
            for label in cur_labels:
                print(f"== {label} == ")
                print(torch.stack(dists[label])[:5].cpu())
            
    # raw_distribution_path : 
    
if __name__ == "__main__":
    main()

