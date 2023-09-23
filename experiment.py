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
#from nn_pruning.patch_coordinator import (
#    SparseTrainingArguments,
#    ModelPatchingCoordinator,
#)
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
    elif config["dev-name"] == 'reweight': config["dev_json"]['reweight'] = 'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl'

    geting_counterfactual_paths(config)
    geting_NIE_paths(config,mode)

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = BertForSequenceClassification.from_pretrained(config["model_name"])
    model = model.to(DEVICE)

    seed = config['seed'] #random.randint(0,10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    output_dir = '../models/recent_baseline/' 
    if os.path.exists(output_dir): 
        output_dir = os.path.join(output_dir, "seed_"+ str(seed))
        print(f'random seed : {seed}')

    # Todo: find all the components used for to clasisfiy our tasks
    # Custom model to be able to custom grad when perform brackpropagation

    # Todo: generalize for every model 
    # using same seed everytime we create HOL and LOL sets 
    experiment_set = ExperimentDataset(config, encode = tokenizer)                            
    dataloader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)
    
    # Todo: test on 
    if config['getting_counterfactual']: collect_output_components(model, config, experiment_set, dataloader, tokenizer, DEVICE) 
    if config['print_config']: print_config(config)
    if not os.path.isfile(save_nie_set_path): get_nie_set_path(config, experiment_set, save_nie_set_path)
    if config['analysis']:  cma_analysis(config, save_nie_set_path = save_nie_set_path, model = model, treatments = mode, tokenizer = tokenizer, experiment_set = experiment_set, DEVICE = DEVICE, DEBUG = True)
    # if config['topk']: print(f"the NIE paths are not available !") if sum(config['is_NIE_exist']) != len(config['is_NIE_exist']) else get_top_k(config, treatments=mode) 
    if config['topk']: get_top_k(config, treatments=mode) 
    if config['embedding_summary']: compute_embedding_set(experiment_set, model, tokenizer, config['label_maps'], DEVICE, config['is_group_by_class'])
    if config['distribution']: get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)
    if config['debias_test']: debias_test(config, model, experiment_set, tokenizer, DEVICE)
    if config['traced']: trace_counterfactual(model, save_nie_set_path, tokenizer, DEVICE, debug)
    if config["diag"]: get_diagnosis(config)
    if config['rank_losses']: rank_losses(config=config, do=mode[0])
    if config['partition_params']: partition_param_train(model, tokenizer, config, mode[0], DEVICE)
    if config['get_condition_inferences']: get_condition_inferences(config, mode[0], model, tokenizer, DEVICE)
    if config['get_inference_based']:  get_inference_based(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, is_load_model= True, is_optimized_set=False)
    if config['traced_params']: trace_optimized_params(model, config, DEVICE, is_load_optimized_model=True)
    if config['test_traced_params']: test_restore_weight(model, config, DEVICE)
    # get_analysis(config)
    
if __name__ == "__main__":
    main()


