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
from my_package.intervention import get_mediators
from my_package.cma_utils import get_component_names
from my_package.cma_utils import get_avg_nie
from my_package.cma_utils import combine_pos
from my_package.intervention import ablation_intervention
#from nn_pruning.patch_coordinator import (
#    SparseTrainingArguments,
#    ModelPatchingCoordinator,
#)


class CounterfactualRepresentation:
    def __init__(self, label_maps, tokenizer, is_group_by_class) -> None:
        
        self.representations = {}
        self.poolers = {}
        self.counter = {}
        
        self.acc = {}
        self.class_acc = {}
        self.confident = {}

        self.label_maps = label_maps
        self.label_remaps = {v:k for k, v in self.label_maps.items()}
        self.tokenizer = tokenizer

        for do in ["High-overlap", "Low-overlap"]:
            if is_group_by_class:
                self.representations[do] = {}
                self.poolers[do] = {}
                self.counter[do] = {}
                # scores
                self.acc[do] = {}
                self.class_acc[do] = {}
                self.confident[do] = {}
            else:
                self.representations[do] = []
                self.poolers[do] = 0
                self.counter[do] = 0
                # scores
                self.acc[do] = []
                self.class_acc[do] = {}
                self.confident[do] = {}
                
                for label_name in self.label_maps.keys():
                    self.class_acc[do][label_name] = []
                    self.confident[do][label_name] = 0

def cma_analysis(config, model_path, method_name, seed, counterfactual_paths, NIE_paths, save_nie_set_path, model, treatments, tokenizer, experiment_set, DEVICE, DEBUG=False):
    # checking model and counterfactual_paths whether it change corresponding to seeds
    mediators = {}
    counter = None
    nie_dataset = None
    nie_dataloader = None
    counter_predictions  = {} 
    layers = config['layers']  if config['computed_all_layers'] else [config['layer']]
    assert len(layers) == 12, f"This doesn't cover all layers"
    # NIE_path = { sorted(path.split('_'),key=len)[1 if config['dataset_name'] == 'qqp' else 0  ]: path for path in NIE_paths} 
    NIE_path = { path.split('/')[-1].split('_')[-3]: path for path in NIE_paths} 
    NIE_path =  NIE_path['all'] if 'all' in NIE_path.keys() else NIE_path[str(config['layer'])]
    print(f"perform Causal Mediation analysis...")
    import copy 
    if model_path is not None: 
        _model = load_model(path= model_path, model=copy.deepcopy(model))
        print(f'Loading CMA model: {model_path}')
    else:
        _model = copy.deepcopy(model)
        _model = _model.to(DEVICE)
        print(f'using original model as input to this function')
    
    with open(save_nie_set_path, 'rb') as handle:
        nie_dataset = pickle.load(handle)
        nie_dataloader = pickle.load(handle)
        print(f"loading nie sets from pickle {save_nie_set_path} !")        
  
    mediators  = get_mediators(_model)
    NIE = get_component_names(config, intervention=True)
    counter = get_component_names(config, intervention=True)
    #shape: cls[seed][component][do][layer][neuron_id] or cls[seed][component][do][layer][class_name][neuron_id]
    cls = get_hidden_representations(config, counterfactual_paths, method_name, layers, config['is_group_by_class'], config['is_averaged_embeddings'])
    
    if config['is_averaged_embeddings']: 
        high_level_intervention(config, nie_dataloader, mediators, cls, NIE, counter ,layers , _model, config['label_maps'], tokenizer, treatments, DEVICE, seed=seed)
    elif config["is_group_by_class"]:
        for label_text in config['label_maps'].keys():
            print(f'********** {label_text} group **********')
            high_level_intervention(config, nie_dataloader[label_text], mediators, cls, NIE, counter ,layers , _model, config['label_maps'], tokenizer, treatments, DEVICE, class_name=label_text, seed=seed)
        
    
    with open(NIE_path, 'wb') as handle: 
        pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving NIE scores into : {NIE_path}')


def get_topk(config, k=None, top_neuron_num=None):
    if config['eval_candidates']: # Using  hyperparameter from config
        if k is not None:
            topk = {'percent': k / 100}
        elif top_neuron_num is not None:
            topk = {'neurons': top_neuron_num}
    else: # ******************** Hyperparameter search ********************
        params  = get_params(config)
        total_neurons = get_num_neurons(config)
        if k is not None: topk = {"percent": (torch.tensor(list(range(1, k+1))) / 100).tolist()}
        if num_top_neurons is not None:
            topk = {"neurons": (torch.tensor(list(range(0, num_top_neurons+1, 5)))).tolist()} 
        else: 
            topk = {'percent': [config['masking_rate']] if config['masking_rate'] else params['percent']}
    return topk


def get_candidate_neurons(config, method_name, NIE_paths, treatments, debug=False):
    # random seed
    seed = config['seed'] 
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else: 
        seed = str(seed)

    # select candidates  based on percentage
    k = config['k']
    top_neuron_num = config['top_neuron_num']
    mode = config['top_neuron_mode']
    print(f'get_candidate_neurons: {mode}')
    # select candidates based on the number of neurons
    num_top_neurons = config['num_top_neurons']
    top_neuron_path = f'../pickles/top_neurons/{method_name}/'
    if not os.path.exists(top_neuron_path): os.mkdir(top_neuron_path)
    top_neurons = {}
    topk = get_topk(config, k=k, top_neuron_num=top_neuron_num)
    key = list(topk.keys())[0]
    layers = config['layers'] if config['computed_all_layers'] else config['layer']
    save_path = None
    from my_package.cma_utils import get_avg_nie
    
    # compute average NIE
    for cur_path in (t:=tqdm(NIE_paths)):
        seed = cur_path.split('/')[3].split('_')[-1]
        do = cur_path.split('/')[-1].split('_')[2 if config['is_averaged_embeddings'] else 3]
        t.set_description(f"{seed}, {do} : {cur_path}")
        NIE, counter, df_nie = get_avg_nie(config, cur_path, layers)
        from my_package.cma_utils import combine_pos
        df_nie['combine_pos'] = df_nie.apply(lambda row: combine_pos(row), axis=1)
        # select candidate group (specific layer or all layers)

        if config['top_neuron_layer'] is not None: df_nie = df_nie[df_nie.Layers == config['top_neuron_layer']]
        
        ranking_nie = {row['combine_pos']: row['NIE'] for index, row in df_nie.iterrows()}
        if not isinstance(topk[key], list): topk[key] = [topk[key]]
        
        for value in topk[key]:
            num_neurons =  value * df_nie.shape[0] if key == 'percent' else value
            num_neurons = int(num_neurons)
            
            print(f"++++++++ Component-Neuron_id: {round(value, 4) if key == 'percent' else num_neurons} neurons :+++++++++")
            if mode == 'random':
                from operator import itemgetter
                cur_neurons =  [(k, v) for k, v in ranking_nie.items()]
                random.shuffle(cur_neurons)
                
                top_neurons[round(value, 4) if key == 'percent' else value] = dict(cur_neurons[:num_neurons])
            elif mode == 'sorted':
                top_neurons[round(value, 4) if key == 'percent' else value] = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])
            
        # ********  write it to pickle *********
        if config['top_neuron_layer'] is not None:
            opt_layer = config['top_neuron_layer'] 
        elif config['computed_all_layers']:
            opt_layer = 'all_layers'

        if config['is_averaged_embeddings']:
            save_path = os.path.join(top_neuron_path, f'{mode}_top_neuron_{seed}_{key}_{do}_{opt_layer}.pickle')
        elif config['is_group_by_class']:
            save_path = os.path.join(top_neuron_path, f'{mode}_top_neuron_{seed}_{key}_{do}_{opt_layer}_class_level.pickle')
            
        with open(save_path, 'wb') as handle:
            pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Done saving {mode} top neurons into pickle! : {save_path}") 
    

def evalutate_counterfactual(experiment_set, dataloader, config, model, tokenizer, label_maps, DEVICE, all_model_paths, DEBUG=False, summarize=False):
    """ To see the difference between High-overlap and Low-overlap score whether our counterfactuals have huge different."""
    from my_package.cma_utils import foward_hooks
    import copy
    import torch.nn.functional as F
    
    # To see the change of probs comparing between High bias and Low bias inputs
    average_all_seed_distributions = {}
    count_num_seed = 0
    distribution = {}

    for seed, model_path in all_model_paths.items():
        if model_path is not None: 
            _model = load_model(path= model_path, model=copy.deepcopy(model))
            print(f'Loading Counterfactual model: {model_path}')
        else:
            _model = copy.deepcopy(model)
            _model = _model.to(DEVICE)
            print(f'using original model as input to this function')
         
        # distribution[seed] = [] if config["is_averaged_embeddings"] else {}
        classifier = Classifier(model=_model)
        # treatments = ['High-overlap'] if config['dataset_name'] == 'fever' else ['High-overlap','Low-overlap']
        treatments =['High-overlap','Low-overlap']
        counterfactuals = CounterfactualRepresentation(label_maps, tokenizer=tokenizer, is_group_by_class=config["is_group_by_class"])
        distribution[seed] = {}
        print(f'*********** {seed} ***********')

        for do in treatments:
            if config["is_averaged_embeddings"]:
                counter, average_representation = foward_hooks(do, tokenizer, dataloader, _model ,DEVICE, eval=config["eval_counterfactuals"],DEBUG=config['DEBUG'])
                out = classifier(average_representation.T).squeeze(dim=0)
                distribution[seed][do] = F.softmax(out, dim=-1).cpu()
                for label_text, label_id in label_maps.items(): print(f'{do}: {label_text}, { distribution[seed][do][label_id]}')
            elif config["is_group_by_class"]:
                distribution[seed][do] = {}
                for group in config['label_maps'].keys():
                    counter, average_representation = foward_hooks(do, tokenizer, dataloader, _model, DEVICE, class_name=group, eval=config["eval_counterfactuals"],DEBUG=config['DEBUG'])
                    out = classifier(average_representation.T).squeeze(dim=0)
                    distribution[seed][do][group] = F.softmax(out, dim=-1).cpu()
                    print(f":{group} Group:")
                    for label_text, label_id in label_maps.items(): print(f'{do}: {label_text}, { distribution[seed][do][group][label_id]}')

def get_embeddings(experiment_set, model, tokenizer, label_maps, DEVICE):
    
    representations = {}
    poolers = {}
    counter = {}
    sentence_representation_path = '../pickles/sentence_representations.pickle'

    LAST_HIDDEN_STATE = -1 
    CLS_TOKEN = 0

    classifier = Classifier(model=model)

    for do in ['High-overlap','Low-overlap']:
            
        if do not in representations.keys():
            representations[do] = {}
            poolers[do] = {}
            counter[do] = {}
    
        for type in ["contradiction","entailment","neutral"]:
        
            if type not in representations[do].keys():
                
                representations[do][type] = []
                poolers[do][type] = 0
                counter[do][type] = 0
        
            representation_loader = DataLoader(experiment_set.sets[do][type],
                                                batch_size = 64,
                                                shuffle = False, 
                                                num_workers=0)
            
            samples = experiment_set.sets[do][type].pair_label.tolist()
            
            nie_dataset = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(samples)]
            
            nie_loader = DataLoader(nie_dataset, batch_size=64)

            for batch_idx, (sentences, labels) in enumerate(tqdm(nie_loader, desc=f"{do} : {type}")):

                premise, hypo = sentences

                pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
                
                inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")

                labels = [label_maps[label] for label in labels]
                
                inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
                counter[do][type] += inputs['input_ids'].shape[0]            

                with torch.no_grad(): 

                    # Todo: generalize to distribution if the storage is enough
                    outputs = model(**inputs, output_hidden_states=True)

                    # (bz, seq_len, hiden_dim)
                    representation = outputs.hidden_states[LAST_HIDDEN_STATE][:,CLS_TOKEN,:].unsqueeze(dim=1)

                    predictions = torch.argmax(F.softmax(classifier(representation), dim=-1), dim=-1)

                    # (bz, seq_len, hidden_dim)
                    representations[do][type].extend(representation) 

    print(f"Averaging sentence representations of CLS across each set")

    # Forward sentence to get distribution
    for do in ['High-overlap','Low-overlap']:
        
        print(f"++++++++++++++++++  {do} ++++++++++++++++++")
        
        for type in ["contradiction","entailment","neutral"]:

            representations[do][type] = torch.stack(representations[do][type], dim=0)
            average_representation = torch.mean(representations[do][type], dim=0 ).unsqueeze(dim=0)

            out = classifier(average_representation).squeeze(dim=0)
            
            cur_distribution = F.softmax(out, dim=-1)

            print(f">>>>>> {type} set")

            print(f"contradiction : {cur_distribution[label_maps['contradiction']]}")
            print(f"entailment : {cur_distribution[label_maps['entailment']]}")
            print(f"neutral : {cur_distribution[label_maps['neutral']]}")


    # with open(sentence_representation_path, 'wb') as handle:
    #     pickle.dump(sent_representations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f"Done saving sentence representations")
    

def get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE):
    
    # Todo: get prediction 
    # 1. from HOL set
    # 2. from NIE set
    
    # dataloader of CLS representation set
    # dataloader of NIE set
    distributions = {"NIE": [], "High-overlap": [], "Low-overlap": []}
    counters = {"NIE": 0, "High-overlap": 0, "Low-overlap": 0}
    label_collectors = {"NIE": () , "High-overlap": (), "Low-overlap": ()}

    distribution_path = '../pickles/distribution.pickle'
    
    dataloader_representation = DataLoader(experiment_set, 
                                         batch_size = 64,
                                         shuffle = False, 
                                         num_workers=0)
    
    
    with open(save_nie_set_path, 'rb') as handle:
        
        dataset_nie = pickle.load(handle)
        dataloader_nie = pickle.load(handle)
        
        print(f"Loading nie sets from pickle {save_nie_set_path} !")
        
    # get distributions of NIE; dont use experiment_set for dataloader; no do variable
    for batch_idx, (sentences, labels) in enumerate(tqdm(dataloader_nie, desc="NIE_DataLoader")):

        premise, hypo = sentences
        
        pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]
        
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
        
        with torch.no_grad(): 

            # Todo: generalize to istribution if the storage is enough
            distributions['NIE'].extend(F.softmax(model(**inputs).logits , dim=-1))

        counters['NIE'] += inputs['input_ids'].shape[0] 
        label_collectors['NIE'] += labels

    # using experiment set to create dataloader
    for batch_idx, (sentences, labels) in enumerate(tqdm(dataloader_representation, desc="representation_DataLoader")):

        for idx, do in enumerate(tqdm(['High-overlap','Low-overlap'], desc="Do-overlap")):

            premise, hypo = sentences[do]

            pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
            
            inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
            
            inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

            with torch.no_grad(): 

                # Todo: generalize to distribution if the storage is enough
                distributions[do].extend(F.softmax(model(**inputs).logits , dim=-1))

            counters[do] += inputs['input_ids'].shape[0] 
            label_collectors[do] += tuple(labels[do])
            
            print(f"labels {batch_idx} : {labels[do]}")

    
    with open(distribution_path, 'wb') as handle:
        pickle.dump(distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(label_collectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done saving distribution into pickle !")

def scaling_nie_scores(config, method_name, NIE_paths, debug=False, mode='sorted') -> pd.DataFrame:
    # select candidates  based on percentage
    k = config['k']
    # select candidates based on the number of neurons
    num_top_neurons = config['num_top_neurons']
    top_neurons = {}
    num_neurons = None
    topk = get_topk(config, k=k, num_top_neurons=num_neurons)
    key = list(topk.keys())[0]
    # rank for NIE
    layers = config['layers'] if config['computed_all_layers'] else config['layer']
    # compute average NIE
    # ranking_nie = {} if config['compute_all_seeds'] else None
    scores = {"Neuron_ids": None, "NIE_scores": None }
    scaler = MinMaxScaler()
    treatment = "High-overlap"  if config['treatment'] else  "Low-overlap"

    for cur_path in (t:=tqdm(NIE_paths)):
        ranking_nie = {}
        with open(cur_path, 'rb') as handle:
            NIE = pickle.load(handle)
            counter = pickle.load(handle)
            print(f"loading NIE : {cur_path}")

        seed = cur_path.split('/')[3].split('_')[-1]
        do = cur_path.split('/')[-1].split('_')[2]
        path = f'../NIE/{method_name}/'
        path = os.path.join(path, "seed_"+ str(seed))
        for layer in layers:
            for component in NIE[do].keys():
                for neuron_id in NIE[do][component][layer].keys():
                    NIE[do][component][layer][neuron_id] = NIE[do][component][layer][neuron_id] / counter[do][component][layer][neuron_id]
                    ranking_nie[(f"L-{layer}-" if config['computed_all_layers'] else "") + component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
        
        # sort whole layers
        if config['computed_all_layers']:
            all_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True))
            if not isinstance(topk[key], list): topk[key] = [topk[key]]
            for value in topk[key]:
                num_neurons =  len(list(all_neurons.keys())) * value if key == 'percent' else value
                num_neurons = int(num_neurons)
                print(f"++++++++ Component-Neuron_id: {round(value, 4) if key == 'percent' else num_neurons} neurons :+++++++++")
                cur_neurons = sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)
                cur_neurons = dict(cur_neurons)
                scores["Neuron_ids"] = list(cur_neurons.keys())
                scores["NIE_scores"] = list(cur_neurons.values())
                df = pd.DataFrame(scores)
                df['NIE_scores'] = df['NIE_scores'].apply(lambda row :  float(row))
                scaler.fit(df['NIE_scores'].to_numpy().reshape(-1, 1))
                transformer = Normalizer().fit(df['NIE_scores'].to_numpy().reshape(-1, 1))
                df['MinMax'] = scaler.transform(df['NIE_scores'].to_numpy().reshape(-1, 1))
                df['Normalize'] = transformer.transform(df['NIE_scores'].to_numpy().reshape(-1, 1))
                df['M_MinMax'] = df['MinMax'].apply(lambda row : 1-row)
                df['M_Normalize'] = df['Normalize'].apply(lambda row : 1-row)
                df['M_NIE_scores'] = df['NIE_scores'].apply(lambda row : 1-row)
                nie_table_path = os.path.join(path, f'nie_table_avg_embeddings_{treatment}_computed_all_layers_.pickle') 
                
                with open(nie_table_path, 'wb') as handle:
                    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Done saving NIE table into {nie_table_path} !")

    return df

def get_top_neurons_layer_each(config, method_name, NIE_paths, treatments, debug=False):
    # random seed
    seed = config['seed'] 
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else: 
        seed = str(seed)
    
    top_neuron_path = f'../pickles/top_neurons/{method_name}/'
    if not os.path.exists(top_neuron_path): os.mkdir(top_neuron_path)

    # select candidates  based on percentage
    k = config['k']
    mode = config['top_neuron_mode']
    num_neurons = None
    topk = get_topk(config, k=k, top_neuron_num=num_neurons)
    key = list(topk.keys())[0]
    layers = config['layers'] if config['computed_all_layers'] else config['layer']
    sorted_local = {}
    

    # compute average NIE
    for cur_path in (t:=tqdm(NIE_paths)):
        seed = cur_path.split('/')[3].split('_')[-1]
        do = cur_path.split('/')[-1].split('_')[2 if config['is_averaged_embeddings'] else 3]
        t.set_description(f"{seed}, {do} : {cur_path}")
        NIE, counter, df_nie = get_avg_nie(config, cur_path, layers)
        df_nie['combine_pos'] = df_nie.apply(lambda row: combine_pos(row), axis=1)
        # df_nie.sort_values(by = ['NIE'], ascending = False)
        # ranking_nie = {row['combine_pos']: row['NIE'] for index, row in df_nie.iterrows()}
        if not isinstance(topk[key], list): topk[key] = [topk[key]]

        for value in topk[key]:
            sorted_local[value] = {}
            for layer in layers:
                sorted_local[value][layer] = df_nie[df_nie.Layers == layer]
                num_neurons =  value * sorted_local[value][layer].shape[0] if key == 'percent' else value
                num_neurons = int(num_neurons)
                sorted_local[value][layer] = {row['combine_pos']: row['NIE'] for index, row in sorted_local[value][layer].iterrows()}
                sorted_local[value][layer] = dict(sorted(sorted_local[value][layer].items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])
                
    if config['is_averaged_embeddings']:
        save_path = os.path.join(top_neuron_path, f'top_neuron_layer_each_{seed}_{key}_{do}_all_layers.pickle')
    elif config['is_group_by_class']:
        save_path = os.path.join(top_neuron_path, f'top_neuron_layer_each_{seed}_{key}_{do}_all_layers_class_level.pickle')
    
    with open(save_path, 'wb') as handle:
        pickle.dump(sorted_local, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done saving top neurons for each layer into pickle!: {save_path}") 

def get_sequential_neurons(config, save_nie_set_path, counterfactual_paths, model_path, model, method_name, NIE_paths, tokenizer, DEVICE, debug=False):
    import copy 
    
    k = config['k']
    mode = config['top_neuron_mode']
    seed = config['seed']
    do = config['treatment']
    INTERVENTION_CLASS = config['intervention_class'][0] 
    label_maps = config['label_maps']
    layers = config['layers'] if config['computed_all_layers'] else config['layer']
    top_neuron_path = f'../pickles/top_neurons/{method_name}/'
    if not os.path.exists(top_neuron_path): os.mkdir(top_neuron_path)
    
    print(f'Intervention Class: {INTERVENTION_CLASS}')
    print(f'Intervention type : {config["intervention_type"]}')
    print(f'Treatment : {config["treatment"]}')
    print(f'Seed : {seed}')
    assert len(NIE_paths) == 1, f'Expect len 1 but found :{len(NIE_paths)}'
   
    topk = get_topk(config, k=k, num_top_neurons=None)
    key = list(topk.keys())[0]
    cur_path = NIE_paths[0]
    NIE, counter, df_nie = get_avg_nie(config, cur_path, layers)
    df_nie['combine_pos'] = df_nie.apply(lambda row: combine_pos(row), axis=1)
    step = df_nie[df_nie.Layers == 0].shape[0] / 5 # follow papers
    step = int(step)
    point_num = 20
    import math
    
    cls = get_hidden_representations(config, counterfactual_paths, method_name, layers, config['is_group_by_class'], config['is_averaged_embeddings'])
    
    if seed is not None:
        print(f'high level intervention seed:{seed}')
        if isinstance(seed, int): seed = str(seed)
        cls = cls[seed]
    else:
        # when using original model
        cls = cls[str(seed)] 
    
    if model_path is not None: 
        _model = load_model(path= model_path, model=copy.deepcopy(model))
        print(f'Loading CMA model: {model_path}')
    else:
        _model = copy.deepcopy(model)
        _model = _model.to(DEVICE)
        print(f'using original model as input to this function')
    
    with open(save_nie_set_path, 'rb') as handle:
        nie_dataset = pickle.load(handle)
        nie_dataloader = pickle.load(handle)
        print(f"loading nie sets from pickle {save_nie_set_path} !")        
  
    mediators  = get_mediators(_model)
    NIE = {}
    
    for pt in range(point_num): 
        neuron_num = int(math.pow(10, pt/5) * 10)
        next_neuron_num = int(math.pow(10, (pt+1) / 5) * 10)
         
        print(f'************')
        for group in [10, 11, 'all']:
            if group != 'all' and  neuron_num > df_nie[df_nie.Layers == 0].shape[0]: continue
            
            candidate_neurons = df_nie[df_nie.Layers == group].copy() if group != 'all' else  df_nie.copy()
            #sorted 
            candidate_neurons = candidate_neurons.sort_values(by=['NIE'], ascending=False)
            candidate_neurons = candidate_neurons.reset_index(drop=True)

            if neuron_num <= candidate_neurons.shape[0] and  next_neuron_num >= candidate_neurons.shape[0]:
                print(f'{neuron_num} -> {candidate_neurons.shape[0]}')
                neuron_num = candidate_neurons.shape[0]
            
            print(f'{group}: {neuron_num} / {df_nie.shape[0]}, {candidate_neurons.shape[0]}')
            if neuron_num not in NIE.keys():
                NIE[neuron_num] = {}
            
            hooks = []
            NIE[neuron_num][group] = ablation_intervention(config, hooks, _model, nie_dataloader, neuron_num, candidate_neurons, step, group, mediators, cls,  label_maps, tokenizer, DEVICE)
            for hook in hooks: hook.remove()
            del hooks

    path = f'../NIE/{method_name}/'
    if not os.path.exists(path): os.mkdir(path) 
    path = os.path.join(path, "seed_"+ str(config['seed'] if seed is None else seed ) )
    if not os.path.exists(path): os.mkdir(path) 
    
    if config['is_averaged_embeddings']:
        group_NIE_path = os.path.join(path, f'ablation_avg_embeddings_{do}_computed_all_layers_.pickle') 
    elif config['is_group_by_class']: 
        group_NIE_path = os.path.join(path, f'ablation_class_level_embeddings_{do}_computed_all_layers_.pickle')

    with open(group_NIE_path, 'wb') as handle: 
        pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saving Ablation NIE into {group_NIE_path} done ! ")


