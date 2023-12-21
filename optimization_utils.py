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
from cma import get_topk
from utils import LayerParams

def initial_partition_params(config, method_name, model, do, collect_param=False, debug=True, seed=None , mode='sorted'):
    """partition parameters used to freeze  and train(bias parameters)"""
    from utils import report_gpu
    component_mappings = {}
    freeze_params = {}
    train_params = {}
    total_params = {}
    seed = config['seed'] if seed is None else seed
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
        if mode == 'sorted':
            path = f'../pickles/top_neurons/{method_name}/top_neuron_{seed}_{key}_{do}_all_layers.pickle' 
        elif mode == 'random':
            path = f'../pickles/top_neurons/{method_name}/random_top_neuron_{seed}_{key}_{do}_all_layers.pickle' 
        layers = config['layers']
    else: 
        path = f'../pickles/top_neurons/{method_name}/top_neuron_{seed}_{key}_{do}_{layer}.pickle'
        layer = config['layer']
    
    # candidate neurons existed bias 
    with open(path, 'rb') as handle: 
        top_neuron = pickle.load(handle) 
        print(f'loading top neurons : {path}')
    
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
                # ************* get component *************
                if 'self' in cur_name:  
                    component = component_mappings[cur_name[-2]]  # to get Q, K, V
                elif 'attention' in cur_name and 'output' in cur_name: 
                    component = component_mappings['attention.output']  
                else:
                    component = component_mappings[cur_name[-3]]
                
                for neuron_id in range(param.data.shape[0]):
                    pos = f'L-{layer_id}-{component}-{neuron_id}'
                    total_params[value][cur_name[-1]][pos] = param.data[neuron_id] if collect_param else None
                    # preparing to restore weight that are not in partition gradients
                    if pos not in list(top_neuron[value].keys()):
                        # used to masking grad
                        freeze_params[value][cur_name[-1]][pos] = param.data[neuron_id] if collect_param else None
                    else:
                        # used to reverse gradient
                        train_params[value][cur_name[-1]][pos] = param.data[neuron_id]  if collect_param else None
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
        encoder_params = [] 
        for layer_id in range(model.config.num_hidden_layers): 
            encoder_params.append(LayerParams(layer_id, len(train_params[value]['weight']), len(freeze_params[value]['weight']) ))
        
        # collect parameters needed to be frozen while perform optmize step
        for pos in list(freeze_params[value]['weight'].keys()):
            layer_id = int(pos.split('-')[1])
            encoder_params[layer_id].append_frozen(pos, {'weight': freeze_params[value]['weight'][pos], 'bias': freeze_params[value]['bias'][pos]})
        
        for pos in list(train_params[value]['weight'].keys()):
            layer_id = int(pos.split('-')[1])
            encoder_params[layer_id].append_train(pos, {'weight': train_params[value]['weight'][pos], 'bias': train_params[value]['bias'][pos]})
        
        for child in ['weight', 'bias']:
            assert len(train_params[value][child])  == len(list(top_neuron[value].keys()))
            assert len(total_params[value][child])  == len(train_params[value][child]) + len(freeze_params[value][child])
            print(f'# {child} train parameters:  {len(train_params[value][child])} ')
            print(f'# {child} freeze parameters: {len(freeze_params[value][child])} ')
            print(f'# {child} total oparameters: {len(train_params[value][child]) + len(freeze_params[value][child])} ')

        from utils import test_layer_params
        test_layer_params(encoder_params, freeze_params, train_params, value)

        restore_path = f'../pickles/restore_weight/{method_name}/'
        if not os.path.exists(restore_path): os.mkdir(restore_path)
        
        for layer in range(len(encoder_params)): 
            cur_restore_path = os.path.join(restore_path, f'masking-{value}')
            if not os.path.exists(cur_restore_path): os.mkdir(cur_restore_path)
            
            if mode == 'sorted':
                cur_restore_path = os.path.join(cur_restore_path,f'{seed}_layer{layer}_collect_param={collect_param}_components.pickle')
            elif mode == 'random':
                cur_restore_path = os.path.join(cur_restore_path,f'{seed}_radom_layer{layer}_collect_param={collect_param}_components.pickle')
            
            with open(cur_restore_path, 'wb') as handle:
                pickle.dump(encoder_params[layer], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"saving {layer}'s components into {cur_restore_path}")
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
    mask = torch.ones_like(grad)
    mask[neuron_ids] = 0
    if DEBUG: print(f'call back masking_grad func: {param_name}, {grad.shape}, {mask[neuron_ids].shape}')
    
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
    mask = torch.ones_like(grad)
    mask[neuron_ids] = -1
    if DEBUG: print(f'call back reverse gradient func: {param_name}, {grad.shape}, {mask[neuron_ids].shape}')
    
    return grad  * mask

def get_advantaged_samples(config, model, seed, metric, LOAD_MODEL_PATH, is_load_model, method_name, device, collect=False):
    # Todo: divide label
    biased_label_maps = config['label_maps']
    main_label_maps = config['label_maps']
    biased_label_remaps = {v:k for k,v in biased_label_maps.items()}
    main_label_remaps   = {v:k for k,v in main_label_maps.items()}
    data_path  = config["data_path"]
    train_data = config["train_data"]
    data_path = os.path.join(data_path, train_data)
    biased_df = pd.read_json(data_path, lines=True)
    seed = str(seed)
    predictions = []
    results = []
    candidated_class = config['candidated_class'][0]

    if collect:
        from data import get_all_model_paths
        all_paths = get_all_model_paths(LOAD_MODEL_PATH)
        path = all_paths[seed]
        if is_load_model:
            from utils import load_model
            model = load_model(path=path, model=model, device=device)
            print(f'Loading model from : {path}')
        else:
            print(f'Using original model')
        # # ************* Biased model **************
        for index, row in biased_df.iterrows():
            prediction =  biased_label_remaps[int(torch.argmax(torch.Tensor(row['bias_probs']), dim=0))]
            predictions.append(prediction)
            # results.append(prediction  == row['gold_label'])
            results.append(prediction  == candidated_class)
        
        biased_df['predictions'] = predictions
        biased_df['results'] = results
        if config['dataset_name'] == 'qqp': biased_df['gold_label'] = biased_df['is_duplicate'].apply(lambda row : biased_label_remaps[row])
        biased_df['gold_label_ids'] = biased_df['gold_label'].apply(lambda row : biased_label_maps[row])
        biased_df['prediction_ids'] = biased_df['predictions'].apply(lambda row : biased_label_maps[row])

        print(f"Bias model acc : {metric.compute(predictions=biased_df['prediction_ids'].tolist() , references=biased_df['gold_label_ids'].tolist() ) }")
        
        for label_name in config['label_maps'].keys():
            current_df = biased_df[biased_df['gold_label'] == label_name]
            print(f"{label_name} acc : {metric.compute(predictions=current_df['prediction_ids'].tolist() , references=current_df['gold_label_ids'].tolist() ) }")
        
        print(f'candidated class : {config["candidated_class"]}') 
        # ************* Main model **************
        from data import CustomDataset
        train_set = CustomDataset(config, label_maps=main_label_maps, data_mode="train_data", is_trained=False)
        train_dataloader = DataLoader(train_set, batch_size = 32, shuffle = False, num_workers=0)
        tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
        norm = nn.Softmax(dim=-1)
        main_model = {}
        for col in ['gold_label', 'sentence1', 'sentence2', 'probs','predictions', 'results']: main_model[col] = []
        count = 0
        for inputs in tqdm(train_dataloader):
            sentences1, sentences2,  labels = inputs
            pair_sentences =  [[sent1, sent2] for sent1, sent2  in zip(sentences1, sentences2) ]
            model_inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
            model_inputs = {k:v.to(device) for k,v in model_inputs.items()}
            with torch.no_grad():
                out = model(**model_inputs)[0]
                cur_probs = norm(out)
                cur_preds = torch.argmax(cur_probs, dim=-1).cpu()
                cur_res = cur_preds == labels
                main_model["sentence1"].extend(sentences1)
                main_model["sentence2"].extend(sentences2)
                main_model["gold_label"].extend(labels)
                main_model["probs"].extend(cur_probs.cpu())
                main_model["predictions"].extend(cur_preds.cpu())
                main_model["results"].extend(cur_res)
        
        main_df = pd.DataFrame.from_dict(main_model) 
        main_df['gold_label'] = main_df['gold_label'].apply(lambda row : int(row))
        main_df['predictions'] = main_df['predictions'].apply(lambda row : int(row))
        print(f"Main model acc : {metric.compute(predictions=main_df['predictions'].tolist() , references=main_df['gold_label'].tolist() ) }")
        path = f'../pickles/advantaged/{config["dataset_name"]}_{method_name}_{seed}_inferences.pickle'
        with open(path, 'wb') as handle: 
            pickle.dump(main_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(biased_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saving to {path} done ! ")
        
        with open(path, 'rb') as handle: 
            main_df = pickle.load(handle)
            biased_df = pickle.load(handle)
            print(f"loading to {path} done ! ")
        
        main_df['results'] = main_df['results'].apply(lambda row: bool(row))
        print(f'bias shape : {biased_df.shape }')
        print(f'main shape : {main_df.shape}')
        # select samples base on main model and bias model inferences
        advantaged  = []
        for idx in range(main_df.shape[0]):
            # if main_df['results'].iloc[idx] ==  False and biased_df['results'].iloc[idx] == True and biased_df['gold_label'].iloc[idx] == candidated_class:
            if main_df['results'].iloc[idx] ==  False and biased_df['results'].iloc[idx] == True:  
                advantaged.append(True)
            else: 
                advantaged.append(False)
        
        advantaged_main = main_df[advantaged]
        advantaged_bias = biased_df[advantaged]
        disadvantaged_main = main_df[list(~np.array(advantaged))] 
        disadvantaged_bias = biased_df[list(~np.array(advantaged))] 
        
        assert advantaged_main.shape[0] == advantaged_bias.shape[0]
        assert disadvantaged_main.shape[0] == disadvantaged_bias.shape[0]

        print(f'#advantaged samples: {advantaged_bias.shape[0]}, #disadvantaged samples: {disadvantaged_bias.shape[0]}')

        bias_probs = []
        for row_idx, row  in advantaged_bias.iterrows():
            bias_probs.append(row['bias_probs'])
        bias_probs = torch.tensor(bias_probs)

        probs = {}
        for idx, row in advantaged_bias.iterrows():
            for label_text in biased_label_maps.keys():
                if label_text not in probs.keys(): probs[label_text] = []
                probs[label_text].append(row['bias_probs'][biased_label_maps[label_text]])
        
        for label_text in probs.keys():
            print(f'{label_text} : {len(probs[label_text])}')
            advantaged_bias[label_text + '_probs'] = probs[label_text]

        print(f'Candidate {candidated_class} probs:')
        print(f"max: {advantaged_bias[candidated_class +'_probs'].max()}")
        print(f"min: {advantaged_bias[candidated_class +'_probs'].min()}")
        path = f'../pickles/advantaged/{config["dataset_name"]}_{method_name}_clean_{seed}_inferences.pickle'
        with open(path, 'wb') as handle: 
            pickle.dump(advantaged_main, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(advantaged_bias, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(disadvantaged_main, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(disadvantaged_bias, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saving to {path} done ! ")
    else:
        path = f'../pickles/advantaged/{config["dataset_name"]}_{method_name}_clean_{seed}_inferences.pickle'
        with open(path, 'rb') as handle: 
            advantaged_main = pickle.load(handle)
            advantaged_bias = pickle.load(handle)
            disadvantaged_main = pickle.load(handle)
            disadvantaged_bias = pickle.load(handle)
            print(f"Loading from {path} done ! ")
        
    for label_text in config['candidated_class']:
        mask = advantaged_bias.gold_label == label_text
        label_prob = label_text + '_probs'
        max = advantaged_bias[mask][label_prob].max()
        min = advantaged_bias[mask][label_prob].min()
        mean = advantaged_bias[mask][label_prob].mean()
        print(f'{label_text}, max:{max}, min:{min}, mean:{mean}')

    # assert len(advantaged_bias.gold_label.unique()) == 1
    # assert candidated_class in advantaged_bias.gold_label.unique()
    print(f'#advantaged samples: {advantaged_bias.shape[0]}, #disadvantaged samples: {disadvantaged_bias.shape[0]}')
    
    return  advantaged_main, advantaged_bias

        
        





    






