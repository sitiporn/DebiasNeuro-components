import pickle
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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from data import get_mediators, get_hidden_representations, EncoderParams, get_specific_component, Dev, group_layer_params
from transformers import AutoTokenizer, BertForSequenceClassification
from functools import partial
from cma import get_topk

def initial_partition_params(config, model, do, counterfactual_paths, dataloader,debug=True):
    """partition candidate parameters used to train main model """
    component_mappings = {}
    freeze_params = {}
    train_params = {}
    total_params = {}
    seed = config['seed']
    k = config['k']
    num_neurons = None
    topk = get_topk(config, k=k, num_top_neurons=num_neurons)
    key = list(topk.keys())[0]
    mediators  = get_mediators(model)
    # dev_set = Dev(config['dev_path'], config['dev_json'])
    # dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else ( [config['masking_rate']] if config['masking_rate'] is not None else list(top_neuron.keys()))
    top_k_mode =  'percent' if config['range_percents'] else ('k' if config['k'] else 'neurons')
    if config['computed_all_layers']:
        path = f'../pickles/top_neurons/top_neuron_{seed}_{key}_{do}_all_layers.pickle'
        layers = config['layers']
    else: 
        path = f'../pickles/top_neurons/top_neuron_{seed}_{key}_{do}_{layer}.pickle'
        layer = config['layer']

    # candidate neurons existed bias 
    with open(path, 'rb') as handle: 
        top_neuron = pickle.load(handle) 
    cls = get_hidden_representations(counterfactual_paths, layers, config['is_group_by_class'], config['is_averaged_embeddings'])
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v
    # unfreeze all parameters
    for param in model.parameters(): param.requires_grad = True
    # select masking_rate : 0.05
    for value in (n:= tqdm(num_neuron_groups)):
        # encoder parameter collectors
        freeze_params[value] = {'weight': {}, 'bias': {}}
        train_params[value]  = {'weight': {}, 'bias': {}}
        total_params[value]  = {'weight': {}, 'bias': {}}
        count_param =  0
        #  if 'encoder' not in splited_name: continue
        #  if 'LayerNorm' in splited_name: continue
        for name, param in model.named_parameters():
            cur_name = name.split('.')
            #  To load and save model's parameters
            if 'encoder' in cur_name and 'LayerNorm' not in cur_name:
                component = None
                layer_id = int(cur_name[3])
                count_param  += param.shape[0]
                if 'self' in cur_name:  
                    component = component_mappings[cur_name[-2]]  # to get Q, K, V
                elif 'attention' in cur_name and 'output' in cur_name: 
                    component = component_mappings['attention.output']  
                else:
                    component = component_mappings[cur_name[-3]]
                for neuron_id in range(param.data.shape[0]):
                    cur_combine = f'L-{layer_id}-{component}-{neuron_id}'
                    total_params[value][cur_name[-1]][cur_combine] = param.data[neuron_id]
                    # preparing to restore weight that are not in partition gradients
                    if cur_combine not in list(top_neuron[value].keys()):
                        freeze_params[value][cur_name[-1]][cur_combine] = param.data[neuron_id]
                    else:
                        train_params[value][cur_name[-1]][cur_combine] = param.data[neuron_id]
            else:
                print(f'freeze whole tensor: {name}')
                param.requires_grad = False
        
        for child in ['weight', 'bias']:
            assert len(train_params[value][child])  == len(list(top_neuron[value].keys()))
            assert len(total_params[value][child])  == len(train_params[value][child]) + len(freeze_params[value][child])
            print(f'# {child} train parameters:  {len(train_params[value][child])} ')
            print(f'# {child} freeze parameters: {len(freeze_params[value][child])} ')
            print(f'# {child} total oparameters: {len(train_params[value][child]) + len(freeze_params[value][child])} ')

        print(f'count_param : {count_param}') 
        for name, param in model.named_parameters(): 
            if 'encoder' in name.split('.') and 'LayerNorm' not in name.split('.'): 
                assert param.requires_grad == True, f' Error : {name}'
            else: 
                assert param.requires_grad == False, f' Error : {name}'
        # Todo: rolling out memory
        layer_param = [] 
        for layer_id in range(model.config.num_hidden_layers): 
            layer_param.append(EncoderParams(layer_id, len(train_params[value]['weight']), len(freeze_params[value]['weight']) ))
        
        # collect parameters needed to be frozen while perform optmize step
        for pos in list(freeze_params[value]['weight'].keys()):
            layer_param[int(pos.split('-')[1])].append_pos(pos, {'weight': freeze_params[value]['weight'][pos], 'bias': freeze_params[value]['bias'][pos]})
        
        restore_path = f'../pickles/restore_weight/'
        for layer in range(len(layer_param)): 
            cur_restore_path = os.path.join(restore_path, f'v-{value}')
            if not os.path.exists(cur_restore_path): os.mkdir(cur_restore_path)
            cur_restore_path = os.path.join(cur_restore_path,f'layer{layer}_components.pickle')
            with open(cur_restore_path, 'wb') as handle:
                pickle.dump(layer_param[layer], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"saving layer {layer}'s components into pickle files")
    return model

def trace_optimized_params(model, config, DEVICE, is_load_optimized_model=False , DEBUG=False):

    value = 0.05 # the percentage of candidate neurons
    trained_epoch = 0
    
    # Tracking 
    real_freeze_param_count= 0
    real_optimized_param_count = 0
    encoder_tensor_param_count = 0
    non_encoder_tensor_param_count = 0
    total_tensor_param_count = 0
    
    debug_count = 0
    count_frozen_whole_encoder_params = 0 
    component_mappings = {}
    restore_path = f'../pickles/restore_weight/'
    restore_path = os.path.join(restore_path, f'v-{value}')
    mediators  = get_mediators(model)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v
    
    LOAD_MODEL_PATH = f'../pickles/models/reweight_model_partition_params_epoch{trained_epoch}.pth'
    NUM_PARAM_TYPES = 2
    
    # load optimized model
    if is_load_optimized_model: 
        print(f'Loading optimized model from {LOAD_MODEL_PATH}')
        model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    else:
        print(f'Using current model')
    
    # original model
    original_model = BertForSequenceClassification.from_pretrained(config["model_name"])
    original_model = original_model.to(DEVICE)

    for param_name_key in (t := tqdm(model.state_dict())):
        splited_name  = param_name_key.split('.')
        param_name = splited_name[-1]
        if 'encoder' in param_name_key and 'LayerNorm' not in param_name_key:
            encoder_tensor_param_count += 1
            optimized_param = model.state_dict().get(param_name_key)
            non_optimized_param = original_model.state_dict().get(param_name_key)
            layer_id, component = get_specific_component(splited_name, component_mappings)
            cur_restore_path = os.path.join(restore_path, f'layer{layer_id}_components.pickle')
            with open(cur_restore_path, 'rb') as handle: layer_frozen_params = pickle.load(handle)
            # bias parameters
            layer_candidated_params = list(layer_frozen_params.params[param_name].keys())
            
            for neuron_id in range(optimized_param.shape[0]):
                cur_neuron = f'{component}-{neuron_id}'
                is_param_kept =  torch.all(abs(optimized_param[neuron_id] - non_optimized_param[neuron_id]) < 1e-8)
                if cur_neuron in layer_candidated_params and is_param_kept: real_freeze_param_count += 1
                elif cur_neuron not in layer_candidated_params and not is_param_kept: real_optimized_param_count += 1
        else:
            non_encoder_tensor_param_count += 1
                    
    print(f'Tensor params in Encoder : {encoder_tensor_param_count}, Outside Encoder : {non_encoder_tensor_param_count}') 
    print(f'===========  Optimized  parameters  ==============')
    print(f'Real optimized value : {real_optimized_param_count / NUM_PARAM_TYPES} , Expected train parameters : { layer_frozen_params.num_train_params}')
    print(f'===========  Frozen  parameters  ==============')
    print(f'Real : {real_freeze_param_count / NUM_PARAM_TYPES} , Expected : { layer_frozen_params.num_freeze_params}')
    print(f'Expected Total parameters : { layer_frozen_params.total_params }')

def test_restore_weight(model, config, DEVICE):
    
    # change weights >> act like optimized model's parameters in Encoders
    for name, param in model.named_parameters(): 
        cur_name = name.split('.')
        if 'encoder' in cur_name and 'LayerNorm' not in cur_name:
            for neuron_id in range(param.shape[0]):
                with torch.no_grad():
                    param[neuron_id] = torch.randn(param[neuron_id].shape)
                    
    # restore weight
    from optimization import restore_original_weight
    model = restore_original_weight(model, DEBUG=False)
    # checker
    trace_optimized_params(model, config, DEVICE, is_load_optimized_model = False)

def masking_grad(neuron_ids:int, param_name:str, DEBUG:bool, grad):
    """
        A callback to function and get executed after backward pass to freeze some parameters of model

        Args:
          neuron_ids: specific set of position of neurons to freeze weight of all inputs
          param_name: a neuron type
          grad: gradient used to be masked
    """
    if DEBUG: print(f'call back masking_grad func : {param_name}, {grad.shape}')
    mask =  torch.ones_like(grad)
    mask[neuron_ids] = 0
    # masking out gradients 
    return grad  * mask

def reverse_grad(neuron_ids:int, param_name:str, DEBUG:bool, grad):
    """
        A callback to function and get executed after backward pass to unlearn some parameters of model

        Args:
          neuron_ids: specific set of positions of neurons to revese grad correspoding to whole inputs
          param_name: a neuron type
          grad: gradient used to be reversed
    """
    if DEBUG: print(f'call back reverse gradient func : {param_name}, {grad.shape}')
    return -grad