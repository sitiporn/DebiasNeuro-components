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
from data import get_mediators, get_hidden_representations, get_specific_component, Dev, group_layer_params 
from transformers import AutoTokenizer, BertForSequenceClassification
from functools import partial
from optimization_utils import masking_grad, reverse_grad, initial_partition_params, trace_optimized_params


def exclude_grad(model, hooks, config, value = 0.05, collect_param=False):
    DEBUG = False
    seed = config['seed']
    component_mappings = {}
    restore_path = f'../pickles/restore_weight/'
    restore_path = os.path.join(restore_path, f'masking-{value}')
    mediators  = get_mediators(model)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v

    #  walking in Encoder's parameters
    for param_name, param in model.named_parameters(): 
        splited_name = param_name.split('.')
        if 'encoder' not in splited_name: continue
        if 'LayerNorm' in splited_name: continue

        child = splited_name[-1]
        layer_id, component = get_specific_component(splited_name, component_mappings) 
        freeze_param_name = splited_name[-1]
        cur_restore_path = os.path.join(restore_path, f'{seed}_layer{layer_id}_collect_param={collect_param}_components.pickle')
       
        with open(cur_restore_path, 'rb') as handle:
            layer_params = pickle.load(handle)
        
        # group by components 
        train_neuron_ids = group_layer_params(layer_params, mode='train')
        frozen_neuron_ids = group_layer_params(layer_params, mode='freeze')

        print(f'************ {param_name} ****************')
        print(f'#train components :{len(train_neuron_ids.keys())}, #frozen components {len(frozen_neuron_ids.keys())}' )
        from optimization import reverse_grad

        if 'dense' in splited_name:
            if child == 'weight': 
                if component in list(train_neuron_ids.keys()): hooks.append(mediators[component](int(layer_id)).dense.weight.register_hook(partial(reverse_grad, train_neuron_ids[component], param_name, DEBUG)))
                if component in list(frozen_neuron_ids.keys()): hooks.append(mediators[component](int(layer_id)).dense.weight.register_hook(partial(masking_grad, frozen_neuron_ids[component], param_name, DEBUG)))
            elif child == 'bias':
                if component in list(train_neuron_ids.keys()):  hooks.append(mediators[component](int(layer_id)).dense.bias.register_hook(partial(reverse_grad, train_neuron_ids[component], param_name, DEBUG)))
                if component in list(frozen_neuron_ids.keys()): hooks.append(mediators[component](int(layer_id)).dense.bias.register_hook(partial(masking_grad, frozen_neuron_ids[component], param_name, DEBUG)))
            print(f'exlude_grad func dense : {param_name}') 
        else: 
            if child == 'weight': 
                if component in list(train_neuron_ids.keys()):  hooks.append(mediators[component](int(layer_id)).weight.register_hook(partial(reverse_grad, train_neuron_ids[component], param_name, DEBUG )))
                if component in list(frozen_neuron_ids.keys()): hooks.append(mediators[component](int(layer_id)).weight.register_hook(partial(masking_grad, frozen_neuron_ids[component], param_name, DEBUG )))
            elif child == 'bias':
                if component in list(train_neuron_ids.keys()):  hooks.append(mediators[component](int(layer_id)).bias.register_hook(partial(reverse_grad, train_neuron_ids[component], param_name, DEBUG)))
                if component in list(frozen_neuron_ids.keys()): hooks.append(mediators[component](int(layer_id)).bias.register_hook(partial(masking_grad, frozen_neuron_ids[component], param_name, DEBUG)))
            print(f'exlude_grad func : {param_name}')

        # masking grad hooks : 144
        # reverse grad hooks : 134
    
    return model, hooks

def restore_original_weight(model, DEBUG = False):
    
    value = 0.05
    count_freeze_params = 0
    component_mappings = {}
    restore_path = f'../pickles/restore_weight/'
    restore_path = os.path.join(restore_path, f'v-{value}')
    mediators  = get_mediators(model)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v

    #  walking in 
    for name, param in model.named_parameters(): 
        splited_name = name.split('.')
        if 'encoder' not in splited_name: continue
        if 'LayerNorm' in splited_name: continue
        # t.set_description(f"{name}")

        layer_id, component = get_specific_component(splited_name, component_mappings) 
        
        freeze_param_name = splited_name[-1]

        cur_restore_path = os.path.join(restore_path, f'layer{layer_id}_components.pickle')
        
        with open(cur_restore_path, 'rb') as handle:
            layer_params = pickle.load(handle)

        # Todo: vectorize accessing model parameters 
        for neuron_id in range(param.shape[0]):
            cur_comb = f'{component}-{neuron_id}' 
            # restore weight after performing optimize freeze param
            if cur_comb in list(layer_params.params[freeze_param_name].keys()):
                # modifying to restore original weight back 
                with torch.no_grad():
                    param[neuron_id] = layer_params.params[freeze_param_name][cur_comb]
                    count_freeze_params += 1
                    
    return model

def partition_param_train(model, tokenizer, config, do, counterfactual_paths, DEVICE, DEBUG=False):
    epochs = 3
    learning_rate = 2e-5
    grad_direction = None # should be matrix to perform elemense wise by sample 
    criterion = nn.CrossEntropyLoss(reduction = 'none') 
    optimizer = Adam(model.parameters(), lr= learning_rate)
    dev_set = Dev(config['dev_path'], config['dev_json'])
    dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)
    model = initial_partition_params(config, model, do) 
    hooks = []
    # when performing back propagation model it seems register o  ?
    model, hooks = exclude_grad(model, hooks=hooks)
    print(f'Epochs : {epochs}, with learning rate at : {learning_rate}')

    if DEBUG: 
        for name, param in model.named_parameters(): 
            if param.requires_grad == False: 
                print(f'freeze params state : {name}')
    
    
    if DEBUG: 
        print(f'Before optimize model {model.bert.pooler.dense.weight[:3, :3]}')
    
    # todo:
    # 2. collect loss for each step
    # 3. plot losses 
    losses = []
    accuracies = []
    
    for epoch in (e:= tqdm(range(epochs))):
        running_loss = 0.0
        
        for batch_idx, (inputs) in  enumerate(b:= tqdm(dev_loader)):

            model.train()
            cur_inputs = {} 

            for idx, (cur_inp, cur_col) in enumerate(zip(inputs, list(dev_set.df.keys()))): cur_inputs[cur_col] = cur_inp

            # get the inputs 
            pair_sentences = [[premise, hypo] for premise, hypo in zip(cur_inputs['sentence1'], cur_inputs['sentence2'])]
            pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
            pair_sentences = {k: v.to(DEVICE) for k,v in pair_sentences.items()}

            # ignore label_ids when running experiment on hans
            label_ids = torch.tensor([config['label_maps'][label] for label in cur_inputs['gold_label']]) if config['dev-name'] != 'hans' else None 
            
            # ignore label_ids when running experiment on hans
            if label_ids is not None: label_ids = label_ids.to(DEVICE)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # Todo: generalize to distribution if the storage is enough
            outs =  model(**pair_sentences, labels= label_ids if config['dev-name'] != 'hans' else None)

            loss = criterion(outs.logits, label_ids)
            test_loss = torch.mean(loss)
            
            scalers = cur_inputs['weight_score'] if config["dev-name"] == 'reweight' else torch.ones_like(loss)
            scalers = scalers.to(DEVICE)

            assert abs(outs.loss - test_loss) < 1e-6
            assert scalers.shape == loss.shape
            
            # loss =  torch.mean(scalers * loss * grad_direction) 
            loss =  torch.mean(scalers * loss) 
            loss.backward()

            optimizer.step()                
            
            trace_optimized_params(model, config, DEVICE, DEBUG=True)

            if DEBUG: print(f'{model.bert.pooler.dense.weight[:3, :3]}')

            # print statistics
            running_loss += loss.item()
            losses.append(loss.item())
            
            if batch_idx % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        SAVE_MODEL_PATH = f'../pickles/models/reweight_model_partition_params_epoch{epoch}.pth'
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f'save model into {SAVE_MODEL_PATH}')
    
    if DEBUG: 
        print(f'After optimize model {model.bert.pooler.dense.weight[:3, :3]}')
        print(f'pooler requires grad {model.bert.pooler.dense.weight.requires_grad}')

    with open(f'../pickles/losses/{config["dev-name"]}.pickle', 'wb') as handle: 
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving losses into pickle files')
