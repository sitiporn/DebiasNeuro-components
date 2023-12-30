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

from my_package.data import ExperimentDataset, Dev, get_conditional_inferences, eval_model_qqp, print_config, FeverDatasetClaimOnly
from my_package.data import rank_losses
from my_package.optimization_utils import initial_partition_params 
from my_package.optimization import partition_param_train, restore_original_weight

from my_package.intervention import intervene, high_level_intervention
from my_package.cma import cma_analysis, evalutate_counterfactual, get_distribution, get_candidate_neurons #get_top_k
from my_package.utils import debias_test
from my_package.cma_utils import get_nie_set_path
import yaml
from my_package.utils import get_num_neurons, get_params, get_diagnosis, load_model
from my_package.data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from my_package.data import get_all_model_paths
from my_package.data import get_masking_value 

def main():

    # ******************** LOAD STUFF ********************
    # config_path = "./configs/masking_representation.yaml"
    config_path = "./configs/pcgu_config_qqp.yaml"
    with open(config_path, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(f'config: {config_path}')
    DEBUG = True
    debug = False # for tracing top counterfactual 
    group_path_by_seed = {}
    # torch.manual_seed(config['seed'])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], padding=True, truncation="max_length")
    model = BertForSequenceClassification.from_pretrained(config["model_name"])
    model = model.to(DEVICE)
    # experiment_set = ExperimentDataset(config, encode = tokenizer)  
    label_maps = {"not_duplicate": 0, "is_duplicate": 1}
    # ******************** PATH ********************

    # LOAD_MODEL_PATH = '../models/pcgu_qqp_baseline/'
    # LOAD_MODEL_PATH = '../models/baseline_qqp_mysplit/'
    # LOAD_MODEL_PATH = '../models/baseline_qqp_mysplit/'
    # LOAD_MODEL_PATH = '../models/poe_qqp_mysplit/' 
    # LOAD_MODEL_PATH = '../models/pcgu_qqp_reweight/'
    #LOAD_MODEL_PATH = '../models/reweight_qqp_mysplit/' 
    # LOAD_MODEL_PATH = '../models/poe_qqp_mysplit/' 
    LOAD_MODEL_PATH = '../models/pcgu_qqp_pred_correct_only_poe_/' 
    # LOAD_MODEL_PATH = '../models/pcgu_qqp_poe/' 
    #../models/pcgu_qqp_poe/' 
    if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)

    # Eval models on test and challenge sets for all seeds
    eval_model_qqp(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model= True, is_optimized_set=False)

    
if __name__ == "__main__":
    main()