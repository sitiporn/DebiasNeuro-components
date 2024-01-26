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
from my_package.utils import  report_gpu
from my_package.cma_utils import collect_counterfactuals, trace_counterfactual, geting_counterfactual_paths, get_single_representation, geting_NIE_paths, Classifier, test_mask
from my_package.optimization_utils import test_restore_weight, trace_optimized_params
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

from my_package.data import ExperimentDataset, Dev, get_conditional_inferences, eval_model, print_config, eval_model_qqp
from my_package.data import rank_losses
from my_package.optimization_utils import initial_partition_params 
from my_package.optimization import partition_param_train, restore_original_weight

from my_package.intervention import intervene, high_level_intervention
from my_package.cma import cma_analysis, evalutate_counterfactual, get_distribution, get_candidate_neurons #get_top_k
from my_package.utils import debias_test
from my_package.cma_utils import get_nie_set
import yaml
from transformers import AutoTokenizer, BertForSequenceClassification
from my_package.utils import get_num_neurons, get_params, get_diagnosis, load_model
from my_package.data import get_analysis 
from my_package.data import get_all_model_paths
from my_package.data import get_masking_value 
from my_package.optimization import intervene_grad


parser = argparse.ArgumentParser()
parser.add_argument("--model_save_path", type=str, default=None, required=True, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--model_load_path", type=str, default=None, required=True, help="The directory where the model checkpoints will be read to train.")
parser.add_argument("--dataset_name", type=str, help="dataset name to train") 
parser.add_argument("--method_name", type=str, help="method to train") 
parser.add_argument("--seed", type=int, default=1548, help="The random seed value")	
parser.add_argument("--random_adv", type=bool,  default=False,  help="The random advanated samples")	
parser.add_argument("--collect_adv", type=bool, default=False,  help="collect advanated samples")	
parser.add_argument("--correct_pred", type=bool, default=True, help="model to filter advantaged samples")	
parser.add_argument("--num_epochs", type=int, default=15, help="Total number of training epochs.")
parser.add_argument("--candidated_class", type=str, help="class used to filter advantaged samples") 
parser.add_argument("--intervention_class", type=str, help="class used to compute NIE scores") 
parser.add_argument("--intervention_type", type=str, help="tye of neuron intervention") 
parser.add_argument("--top_neuron_mode", type=str, help="mode select neurons to perform gradients unlearning") 
parser.add_argument("--grad_direction", type=str, help="moving gradient directoin used to learn or unlearn") 
parser.add_argument("--k", type=int, default=5, help="the percentage of total number of neurons") 
parser.add_argument("--compare_frozen_weight", type=bool, default=True, help="compare weight to reference model to restore back to model during training at each step")	
parser.add_argument("--is_averaged_embeddings", type=bool, default=True, help="Average representation across samples")	

args = parser.parse_args()
print(args)
# ******************** LOAD STUFF ********************
# config_path = "./configs/experiment_config.yaml"
if args.dataset_name == 'mnli':
    config_path = "./configs/pcgu_config.yaml"
elif args.dataset_name == 'qqp':
    config_path = "configs/pcgu_config_qqp.yaml"
elif args.dataset_name == 'fever':
    config_path = "./configs/pcgu_config_fever.yaml"

with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(f'config: {config_path}')

LOAD_MODEL_PATH = args.model_load_path
output_dir = args.model_save_path
method_name =  args.method_name
config['seed'] = args.seed 
method_name = args.method_name 
config['dataset_name'] = args.dataset_name
config['random_adv'] = args.random_adv
config['collect_adv'] = args.collect_adv
config['correct_pred'] = args.correct_pred
config['num_epochs'] =  args.num_epochs
config['candidated_class'] =  [args.candidated_class]
config['intervention_class'] =  [args.intervention_class]
config['intervention_type']  = args.intervention_type
config['top_neuron_mode'] = args.top_neuron_mode
config['grad_direction'] = args.grad_direction
config['k'] = args.k
config['compare_frozen_weight'] = args.compare_frozen_weight
config['is_averaged_embeddings'] = args.is_averaged_embeddings

DEBUG = True
debug = False # for tracing top counterfactual 
group_path_by_seed = {}
# torch.manual_seed(config['seed'])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(config['label_maps'].keys()))
model = model.to(DEVICE)
experiment_set = ExperimentDataset(config, encode = tokenizer, seed=config['seed'], dataset_name=config['dataset_name'])                            
dataloader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)

# ******************** PATH ********************
save_nie_set_path = f'../pickles/class_level_nie_{config["num_samples"]}_samples.pickle' if config['is_group_by_class'] else f'../pickles/nie_{config["num_samples"]}_samples.pickle'

LOAD_MODEL_PATH = '../models/recent_baseline/' 
method_name =  f'{config["intervention_type"]}_intervention_{LOAD_MODEL_PATH.split("/")[-2]}'
LOAD_MODEL_PATH = output_dir


NIE_paths = []
if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
if not os.path.isfile(save_nie_set_path): get_nie_set (config, experiment_set, save_nie_set_path)
# model_path = config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])] 
# ******************** Identifying Bias: Causal Mediation Analysis ********************
mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
config['treatment'] = mode[0]
print(f'Counterfactual type: {mode}')
print(f'Intervention type : {config["intervention_type"]}')


from my_package.utils import compare_frozen_weight, prunning_biased_neurons

if config["compute_all_seeds"]:
    for seed, model_path in all_model_paths.items():
        # path to save
        # Done checking path 
        counterfactual_paths, _ = geting_counterfactual_paths(config, method_name=method_name)
        NIE_path, _ = geting_NIE_paths(config, method_name, mode, seed=seed)
        NIE_paths.extend(NIE_path)
        if config['getting_counterfactual']: 
            # Done checking model counterfactual_path and specific model
            collect_counterfactuals(model, model_path, seed, counterfactual_paths, config, experiment_set, dataloader, tokenizer, DEVICE=DEVICE) 
else:
    # path to save counterfactuals 
    counterfactual_paths, _ = geting_counterfactual_paths(config, method_name)
    # path to save NIE scores
    NIE_paths, _ = geting_NIE_paths(config, method_name, mode)
    print(f'Loading path for single at seed:{config["seed"]}, layer: {config["layer"]}')
    for path in counterfactual_paths: print(f"{sorted(path.split('_'), key=len)[0]}: {path}")
    print(f'NIE_paths: {NIE_paths}')
    if config['getting_counterfactual']: 
        # Done checking model counterfactual_path and specific model
        seed = config['seed']
        collect_counterfactuals(model, model_path, seed, counterfactual_paths, config, experiment_set, dataloader, tokenizer, DEVICE=DEVICE) 

from my_package.data import get_condition_inference_hans_result


if config['compute_nie_scores']:  cma_analysis(config, 
                                                config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])], 
                                                config['seed'], 
                                                counterfactual_paths, 
                                                NIE_paths, 
                                                save_nie_set_path = save_nie_set_path, 
                                                model = model, 
                                                treatments = mode, 
                                                tokenizer = tokenizer, 
                                                experiment_set = experiment_set, 
                                                DEVICE = DEVICE, 
                                                DEBUG = True)
    
# if config['get_candidate_neurons']: get_candidate_neurons(config, method_name, NIE_paths, treatments=mode, debug=False) 

if config['distribution']: get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)
if config['rank_losses']: rank_losses(config=config, do=mode[0])
# if config['topk']: print(f"the NIE paths are not available !") if sum(config['is_NIE_exist']) != len(config['is_NIE_exist']) else get_top_k(config, treatments=mode) 
# ******************** Debias ********************
if config["dev-name"] == 'mismatched': config["dev_json"]['mismatched'] = 'multinli_1.0_dev_mismatched.jsonl'
elif config["dev-name"] == 'hans': config["dev_json"]['hans'] = 'heuristics_evaluation_set.jsonl' 
elif config["dev-name"] == 'matched': config["dev_json"]['matched'] = 'multinli_1.0_dev_matched.jsonl'
elif config["dev-name"] == 'reweight': config["dev_json"]['reweight'] = 'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl'
# find hyperparameters for soft masking method
if config['get_condition_inferences']: get_conditional_inferences(config, mode[0], model_path, model, counterfactual_paths, tokenizer, DEVICE, debug = False)
if config['get_condition_hans_result']: get_condition_inference_hans_result( config, model, model_path)
if config['get_masking_value']: get_masking_value(config=config)
# PCGU: optimization 
if config['partition_params']: partition_param_train(model, tokenizer, config, mode[0], counterfactual_paths, DEVICE)
# ******************** test  stuff ********************
if config['eval_model']: 
    if args.dataset_name == 'mnli':
        eval_model(model, NIE_paths, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, method_name=method_name, is_load_model= True, is_optimized_set=False)
    elif args.dataset_name == 'qqp':
        eval_model_qqp(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model= True, is_optimized_set=False)




