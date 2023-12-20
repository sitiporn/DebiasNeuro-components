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
from utils import get_params, get_num_neurons, load_model
from data import get_all_model_paths
from cma_utils import get_overlap_thresholds, group_by_treatment, test_mask, Classifier, get_hidden_representations
from cma_utils import geting_counterfactual_paths, get_single_representation, geting_NIE_paths, collect_counterfactuals
from utils import report_gpu
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
from pprint import pprint
from data import ExperimentDataset
from intervention import intervene, high_level_intervention
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

# from transformers import AutoTokenizer, BertForSequenceClassification
# from data import get_mediators, get_hidden_representations, get_specific_component, Dev, group_layer_params 
# import yaml
# output_dir = '../models/outsidev2_grad_unlearning_poe/' 
# LOAD_MODEL_PATH = '../models/poe2/' 
# method_name = 'poe2'
# value = 0.05
# restore_path = f'../pickles/restore_weight/{method_name}/'
# restore_path = os.path.join(restore_path, f'masking-{value}')

# component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
# component_mappings = {}

# config_path = "./configs/pcgu_config.yaml"
# with open(config_path, "r") as yamlfile:
#     config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# ref_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
# trained_model_paths = get_all_model_paths(output_dir)

# ref_model_path = config['seed'] if config['seed'] is None else ref_model_paths[str(config['seed'])] 
# trianed_model_path = config['seed'] if config['seed'] is None else trained_model_paths[str(config['seed'])] 

# label_maps = config['label_maps'] 
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
# reference_model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))

# reference_model = load_model(path=ref_model_path, model=reference_model, device=DEVICE)
# model = load_model(path=trianed_model_path, model=model, device=DEVICE)

# mediators  = get_mediators(reference_model)
# for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v

# for (ref_name, ref_param), (trained_name, trained_param) in zip(reference_model.named_parameters(), model.named_parameters()):
#     splited_name = ref_name.split('.')
#     if 'layer' not in splited_name: continue
    
#     layer_id, component = get_specific_component(splited_name, component_mappings) 
#     freeze_param_name = splited_name[-1]
#     # Custom adam optmizer
#     cur_restore_path = None
#     if config["top_neuron_mode"] == 'sorted':
#         cur_restore_path = os.path.join(restore_path, f"{config['seed']}_layer{layer_id}_collect_param={config['collect_param']}_components.pickle")
#     elif config["top_neuron_mode"] == 'random':
#         cur_restore_path = os.path.join(restore_path, f"{config['seed']}_radom_layer{layer_id}_collect_param={config['collect_param']}_components.pickle")
    
#     # f'{self.seed}_layer{layer_id}_collect_param={self.collect_param}_components.pickle'
#     with open(cur_restore_path, 'rb') as handle: layer_params = pickle.load(handle)
#     frozen_neuron_ids = group_layer_params(layer_params, mode='freeze')
#     train_neuron_ids = group_layer_params(layer_params, mode='train')

#     neuron_ids = []
#     neuron_ids += frozen_neuron_ids[component] if component in frozen_neuron_ids.keys() else []

#     ref_param[neuron_ids] == trained_param[neuron_ids]
    


 