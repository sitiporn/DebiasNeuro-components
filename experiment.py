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
from utils import  report_gpu
from cma_utils import collect_counterfactuals, trace_counterfactual, geting_counterfactual_paths, get_single_representation, geting_NIE_paths, Classifier, test_mask
from optimization_utils import test_restore_weight, trace_optimized_params
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

from data import ExperimentDataset, Dev, get_conditional_inferences, eval_model, print_config
from data import rank_losses
from optimization_utils import initial_partition_params 
from optimization import partition_param_train, restore_original_weight

from intervention import intervene, high_level_intervention
from cma import cma_analysis, evalutate_counterfactual, get_distribution, get_candidate_neurons #get_top_k
from utils import debias_test
from cma_utils import get_nie_set_path
import yaml
from utils import get_num_neurons, get_params, get_diagnosis, load_model
from data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from data import get_all_model_paths
from data import get_masking_value 
from optimization import exclude_grad
from data import get_condition_inference_scores

def main():

    # ******************** LOAD STUFF ********************
    config_path = "./configs/tiny_masking_rep.yaml"
    with open(config_path, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(f'config: {config_path}')
    DEBUG = True
    debug = False # for tracing top counterfactual 
    group_path_by_seed = {}
    # torch.manual_seed(config['seed'])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = BertForSequenceClassification.from_pretrained(config["model_name"])
    model = model.to(DEVICE)
    experiment_set = ExperimentDataset(config, encode = tokenizer)                            
    dataloader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)
    # ******************** PATH ********************
    save_nie_set_path = f'../pickles/class_level_nie_{config["num_samples"]}_samples.pickle' if config['is_group_by_class'] else f'../pickles/nie_{config["num_samples"]}_samples.pickle'
    # LOAD_MODEL_PATH = '../models/recent_baseline/'
    # LOAD_MODEL_PATH = '../models/reweight2/'
    LOAD_MODEL_PATH = '../models/poe2/'
    # LOAD_MODEL_PATH = '../models/developing_baseline/'
    # method_name =  'recent_baseline' 
    # method_name =  'reweight2' 
    method_name =  'poe2' 
    
    NIE_paths = []
    counterfactual_paths = []
    if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
    if not os.path.isfile(save_nie_set_path): get_nie_set_path(config, experiment_set, save_nie_set_path)
    model_path = config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])] 
    # ******************** Identifying Bias: Causal Mediation Analysis ********************
    mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
    print(f'Counterfactual type: {mode}')
    print(f'Intervention type : {config["intervention_type"]}')
    print(f'current model path : {model_path}')

    if config['eval_counterfactual'] and config["compute_all_seeds"]:
        for seed, model_path in all_model_paths.items():
            # see the result of the counterfactual of modifying proportional bias
            evalutate_counterfactual(experiment_set, config, model, tokenizer, config['label_maps'], DEVICE, config['is_group_by_class'], seed=seed,model_path=model_path, summarize=True)
    if config["compute_all_seeds"]:
        for seed, model_path in all_model_paths.items():
            # path to save
            # Done checking path 
            counterfactual_path, _ = geting_counterfactual_paths(config, seed=seed, method_name=method_name)
            NIE_path, _ = geting_NIE_paths(config, method_name, mode, seed=seed)
            NIE_paths.extend(NIE_path)
            counterfactual_paths.extend(counterfactual_path)
            if config['getting_counterfactual']: 
                # Done checking model counterfactual_path and specific model
                collect_counterfactuals(model, model_path, seed, counterfactual_path, config, experiment_set, dataloader, tokenizer, DEVICE=DEVICE) 
    else:
        # path to save counterfactuals 
        counterfactual_path, _ = geting_counterfactual_paths(config, method_name=method_name)
        # path to save NIE scores
        NIE_paths, _ = geting_NIE_paths(config, method_name, mode)
        print(f'Loading path for single at seed:{config["seed"]}, layer: {config["layer"]}')
        for path in counterfactual_path: print(f"{sorted(path.split('_'), key=len)[0]}: {path}")
        print(f'NIE_paths: {NIE_paths}')
        counterfactual_paths.extend(counterfactual_path)
        if config['getting_counterfactual']: 
            # Done checking model counterfactual_path and specific model
            seed = config['seed']
            collect_counterfactuals(model, model_path, seed, counterfactual_path, config, experiment_set, dataloader, tokenizer, DEVICE=DEVICE) 

    
    if config['compute_nie_scores']:  cma_analysis(config, 
                                                  model_path,
                                                  config['seed'], 
                                                  counterfactual_path, 
                                                  NIE_paths, 
                                                  save_nie_set_path = save_nie_set_path, 
                                                  model = model, 
                                                  treatments = mode, 
                                                  tokenizer = tokenizer, 
                                                  experiment_set = experiment_set, 
                                                  DEVICE = DEVICE, 
                                                  DEBUG = True)

    from data import masking_representation_exp
    LOAD_MODEL_PATH = '../models/recent_baseline/'     
    
    if config['get_candidate_neurons']: get_candidate_neurons(config, NIE_paths, treatments=mode, debug=False) 
    if config['distribution']: get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)
    if config['rank_losses']: rank_losses(config=config, do=mode[0])
    # if config['topk']: print(f"the NIE paths are not available !") if sum(config['is_NIE_exist']) != len(config['is_NIE_exist']) else get_top_k(config, treatments=mode) 
    # ******************** Debias ********************
    if config["dev-name"] == 'mismatched': config["dev_json"]['mismatched'] = 'multinli_1.0_dev_mismatched.jsonl'
    elif config["dev-name"] == 'hans': config["dev_json"]['hans'] = 'heuristics_evaluation_set.jsonl' 
    elif config["dev-name"] == 'matched': config["dev_json"]['matched'] = 'multinli_1.0_dev_matched.jsonl'
    elif config["dev-name"] == 'reweight': config["dev_json"]['reweight'] = 'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl'
    # find hyperparameters for soft masking method
    masking_representation_exp(config, model, method_name, experiment_set, dataloader, NIE_paths, LOAD_MODEL_PATH, counterfactual_paths, tokenizer, DEVICE, is_load_model=True)
    if config['get_condition_inferences']: get_conditional_inferences(config, mode[0], model_path, model, counterfactual_path, tokenizer, DEVICE, debug = False)
    if config['get_condition_inference_scores']: get_condition_inference_scores(config, model, model_path)
    if config['get_masking_value']: get_masking_value(config=config)
    # PCGU: optimization 
    if config['partition_params']: partition_param_train(model, tokenizer, config, mode[0], counterfactual_path, DEVICE)
    # ******************** test  stuff ********************
    # Eval models on test and challenge sets for all seeds
    if config['eval_model']: eval_model(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model= False, is_optimized_set=False)
    if config['traced']: trace_counterfactual(model, save_nie_set_path, tokenizer, DEVICE, debug)
    if config['traced_params']: trace_optimized_params(model, config, DEVICE, is_load_optimized_model=True)
    if config["diag"]: get_diagnosis(config)
    if config['test_traced_params']: test_restore_weight(model, config, DEVICE)
    if config['debias_test']: debias_test(config, model, experiment_set, tokenizer, DEVICE)
    # get_analysis(config)
    
if __name__ == "__main__":
    main()


