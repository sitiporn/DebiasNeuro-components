import os
import os.path
import pandas as pd
import random
import pickle
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from my_package.utils import get_params, get_num_neurons, load_model
from my_package.data import get_all_model_paths
from my_package.cma_utils import get_overlap_thresholds, group_by_treatment, test_mask, Classifier, get_hidden_representations
from my_package.cma_utils import geting_counterfactual_paths, get_single_representation, geting_NIE_paths, collect_counterfactuals
from my_package.utils import report_gpu
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
from pprint import pprint
from my_package.data import ExperimentDataset
from my_package.intervention import intervene, high_level_intervention
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer


"""
../pickles/top_neurons/poe2/
-rwxrwxrwx 1 sitipornl sitipornl 1176249 Dec 14 18:01 random_top_neuron_409_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176256 Dec 14 17:58 random_top_neuron_3990_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1177177 Dec 14 17:49 random_top_neuron_3785_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176309 Dec 14 17:49 random_top_neuron_3099_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176605 Dec 14 17:14 random_top_neuron_1548_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1182514 Nov  7 08:46 top_neuron_409_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1185039 Nov  7 08:45 top_neuron_3990_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1185036 Nov  7 08:45 top_neuron_3099_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1182842 Nov  7 08:44 top_neuron_3785_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1185289 Nov  7 08:44 top_neuron_1548_percent_High-overlap_all_layers.pickle
"""

"""
-rwxrwxrwx 1 sitipornl sitipornl 1176705 Dec 14 21:00 top_neuron_3785_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176828 Dec 14 21:00 top_neuron_3099_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176890 Dec 14 20:59 top_neuron_1548_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176719 Dec 14 20:59 top_neuron_409_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176805 Dec 14 20:59 top_neuron_3990_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176155 Dec 14 11:23 random_top_neuron_409_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176291 Dec 14 11:22 random_top_neuron_3990_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176490 Dec 14 11:22 random_top_neuron_3785_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176073 Dec 14 11:20 random_top_neuron_3099_percent_High-overlap_all_layers.pickle
-rwxrwxrwx 1 sitipornl sitipornl 1176340 Dec 14 11:12 random_top_neuron_1548_percent_High-overlap_all_layers.pickle
1. top neuron (CMA) == random neuron
2. select neurons gradient pass through?
3. without select neurons gradient pass or not?
"""
k = 0.05

method_names = ['recent_baseline', 'poe2']
seeds = [1548, 3099, 3785, 3990, 409] 


# for method_name in method_names: 
# print(f'*********** {method_name} ***********')
for seed in seeds:

    weaken_top_neuron_path =  f'../pickles/top_neurons/recent_baseline/top_neuron_{seed}_percent_High-overlap_all_layers.pickle'
    replace_top_neuron_path =  f'../pickles/top_neurons/replace_intervention_recent_baseline/top_neuron_{seed}_percent_High-overlap_all_layers.pickle'
    # rand_neuron_path = f'../pickles/top_neurons/{method_name}/random_top_neuron_{seed}_percent_High-overlap_all_layers.pickle'

    # with open(weaken_top_neuron_path, 'rb') as handle: top_neuron = pickle.load(handle)
    # with open(replace_top_neuron_path, 'rb') as handle: rand_neuron = pickle.load(handle)
    with open(replace_top_neuron_path , 'rb') as handle: top_neuron = pickle.load(handle)
    with open(weaken_top_neuron_path, 'rb') as handle: rand_neuron = pickle.load(handle)

    top_neuron  = sorted(top_neuron[k].items() , key=operator.itemgetter(1), reverse=True)
    top_neuron_keys  = list(dict(top_neuron).keys())
    rand_neuron = sorted(rand_neuron[k].items() , key=operator.itemgetter(1), reverse=True)
    rand_neuron_keys = list(dict(rand_neuron).keys())
    
    count = 0
    for t in top_neuron_keys:
        if t in rand_neuron_keys: count+=1
    
    print(f'seed:{seed}, #overlap count: {count}/{len(top_neuron)}')

from transformers import AutoTokenizer, BertForSequenceClassification
from my_package.data import get_mediators, get_hidden_representations, get_specific_component, Dev, group_layer_params 
# import yaml
# output_dir = '../models/outsidev2_grad_unlearning_poe/' 
# LOAD_MODEL_PATH = '../models/poe2/' 
# method_name = 'poe2'
# value = 0.05


 