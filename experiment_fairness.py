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
from my_package.data import rank_losses, eval_model_fever
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
parser.add_argument("--model_load_path", type=str, default=None, required=True, help="The directory where the model checkpoints will be read to train.")
parser.add_argument("--dataset_name", type=str, help="dataset name to train") 
parser.add_argument("--method_name", type=str, help="method to train") 
parser.add_argument("--seed", type=int, default=1548, help="The random seed value")	

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
method_name =  args.method_name
config['seed'] = args.seed 
config['dataset_name'] = args.dataset_name


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(config['label_maps'].keys()))
model = model.to(DEVICE)

# ******************** PATH ********************

"""
ACC: 
1.MNLI: 
  |Entailment - Contradiction|, |Entailment -Neutral|  แล้วเอามาเฉีล่ย
  1.1 HANS   |Entailment-Nonentailment|
2. FEVER: support vs nonsupport

MAF1: 
3. QQP: paraphrase vs nonparaphrase 
    
Baseline: MNLI, FEVER, QQP
PCGU:
"""

from my_package.fairness import eval_fairness_mnli, eval_fairness_qqp, eval_fairness_fever

if args.dataset_name == 'mnli':
    eval_fairness_mnli(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, method_name=method_name, is_load_model= True, is_optimized_set=False)
elif args.dataset_name == 'qqp':
    eval_fairness_qqp(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model= True, is_optimized_set=False)
elif args.dataset_name == 'fever':
    eval_fairness_fever(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model= True, is_optimized_set=False)
