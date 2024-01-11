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

from data import ExperimentDataset, Dev, get_conditional_inferences, eval_model, print_config, FeverDatasetClaimOnly
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
# from optimization import exclude_grad

def main():

    # ******************** LOAD STUFF ********************
    # config_path = "./configs/masking_representation.yaml"
    config_path = "./configs/experiment_config_re.yaml"
    with open(config_path, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print(f'config: {config_path}')
    DEBUG = True
    debug = False # for tracing top counterfactual 
    group_path_by_seed = {}
    # torch.manual_seed(config['seed'])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'],padding=True, truncation="max_length")
    model = BertForSequenceClassification.from_pretrained(config["model_name"])
    model = model.to(DEVICE)
    # experiment_set = ExperimentDataset(config, encode = tokenizer)  
    label_maps = {"entailment": 0, "contradiction": 1, "neutral": 2}
    # ******************** PATH ********************

    LOAD_MODEL_PATH = '../models/poe_clark/'
    if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
 
    


    # Eval models on test and challenge sets for all seeds
    eval_model(model, config=config,tokenizer=tokenizer,DEVICE=DEVICE, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model= True, is_optimized_set=False)

    
if __name__ == "__main__":
    main()


