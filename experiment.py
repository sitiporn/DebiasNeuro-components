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
from data import ExperimentDataset, Dev, get_condition_inferences, eval_model, print_config, trace_optimized_params
from data import rank_losses, initial_partition_params, restore_original_weight, partition_param_train
from intervention import intervene, high_level_intervention
from cma import cma_analysis, evalutate_counterfactual, get_distribution, get_top_k
from utils import debias_test
from cma_utils import get_nie_set_path
import yaml
from utils import get_num_neurons, get_params, get_diagnosis, load_model
from data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from data import exclude_grad, get_all_model_paths

def main():

    # ******************** LOAD STUFF ********************
    with open("experiment_config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
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
    LOAD_MODEL_PATH = '../models/recent_baseline/'
    if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
    if not os.path.isfile(save_nie_set_path): get_nie_set_path(config, experiment_set, save_nie_set_path)
    # ******************** Identifying Bias: Causal Mediation Analysis ********************
    mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
    print(f'Counterfactual type: {mode}')
    print(f'Intervention type : {config["intervention_type"]}')
    if config['eval_counterfactual'] and config["compute_all_seeds"]:
        for seed, model_path in all_model_paths.items():
            # see the result of the counterfactual of modifying proportional bias
            evalutate_counterfactual(experiment_set, config, model, tokenizer, config['label_maps'], DEVICE, config['is_group_by_class'], seed=seed,model_path=model_path, summarize=True)
    if config["compute_all_seeds"]:
        for seed, model_path in all_model_paths.items():
            # path to save
            # Done checking path 
            counterfactual_paths, _ = geting_counterfactual_paths(config, seed=seed)
            NIE_paths, _ = geting_NIE_paths(config, mode, seed=seed)
            if config['getting_counterfactual']: 
                # Done checking model counterfactual_path and specific model
                collect_counterfactuals(model, model_path, seed, counterfactual_paths, config, experiment_set, dataloader, tokenizer, DEVICE=DEVICE) 
    else:
        # path to save counterfactuals 
        counterfactual_paths, _ = geting_counterfactual_paths(config)
        # path to save NIE scores
        NIE_paths, _ = geting_NIE_paths(config, mode)
        print(f'Loading path for single at seed:{config["seed"]}, layer: {config["layer"]}')
        for path in counterfactual_paths: print(f"{sorted(path.split('_'), key=len)[0]}: {path}")
        print(f'NIE_paths: {NIE_paths}')
    # dont forget to select mode eg. High or Low overlap
    # recheck intervention type
    if config['compute_nie_scores']:  cma_analysis(config, all_model_paths[str(config['seed'])], config['seed'], counterfactual_paths, NIE_paths, save_nie_set_path = save_nie_set_path, model = model, treatments = mode, tokenizer = tokenizer, experiment_set = experiment_set, DEVICE = DEVICE, DEBUG = True)
    if config['topk']: get_top_k(config, treatments=mode) 
    if config['distribution']: get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)
    if config['rank_losses']: rank_losses(config=config, do=mode[0])
    # eval models on test and challenge sets for all seeds
    if config['eval_model']: eval_model(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, is_load_model= True, is_optimized_set=False)
    # if config['topk']: print(f"the NIE paths are not available !") if sum(config['is_NIE_exist']) != len(config['is_NIE_exist']) else get_top_k(config, treatments=mode) 
    # ******************** Unlearn Bias ********************
    if config['partition_params']: partition_param_train(model, tokenizer, config, mode[0], DEVICE)
    # Tune hyperparameters for soft masking method
    if config['get_condition_inferences']: get_condition_inferences(config, mode[0], model, tokenizer, DEVICE)
    # ******************** test  stuff ********************
    if config['traced']: trace_counterfactual(model, save_nie_set_path, tokenizer, DEVICE, debug)
    if config['traced_params']: trace_optimized_params(model, config, DEVICE, is_load_optimized_model=True)
    if config["diag"]: get_diagnosis(config)
    if config['test_traced_params']: test_restore_weight(model, config, DEVICE)
    if config['debias_test']: debias_test(config, model, experiment_set, tokenizer, DEVICE)
    # get_analysis(config)
    
if __name__ == "__main__":
    main()


