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
from pprint import pprint
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)

from utils import get_overlap_thresholds, group_by_treatment, get_hidden_representations
from intervention import Intervention, neuron_intervention, get_mediators
from utils import get_ans, compute_acc


class ExperimentDataset(Dataset):
    def __init__(self, config, encode, DEBUG=False) -> None: 
        
        data_path =  config['dev_path']
        json_file =  config['exp_json']
        upper_bound = config['upper_bound']
        lower_bound = config['lower_bound'] 
        is_group_by_class = config['is_group_by_class']
        num_samples =  config['num_samples']
        
        # combine these two set
        self.encode = encode

        self.premises = {}
        self.hypothesises = {}
        self.labels = {}
        self.intervention = {}
        pair_and_label = []
        self.is_group_by_class = is_group_by_class
        
        self.sets = {"High-overlap": {}, "Low-overlap": {} } 
        nums = {"High-overlap": {}, "Low-overlap": {} }
        
        torch.manual_seed(42)
        
        data_path = os.path.join(data_path, json_file)

        self.df = pd.read_json(data_path, lines=True)

        if '-' in self.df.gold_label.unique():
            self.df = self.df[self.df.gold_label != '-'].reset_index(drop=True)

        if DEBUG: print(self.df.columns)

        for i in range(len(self.df)):
            pair_and_label.append(
                (self.df['sentence1'][i], self.df['sentence2'][i], self.df['gold_label'][i]))

        self.df['pair_label'] = pair_and_label
        
        thresholds = get_overlap_thresholds(self.df, upper_bound, lower_bound)

        # get HOL and LOL set
        self.df['Treatment'] = self.df.apply(lambda row: group_by_treatment(
            thresholds, row.overlap_scores, row.gold_label), axis=1)

        print(f"== statistic ==")
        pprint(thresholds)
        
        self.df_exp_set = {"High-overlap": self.get_high_shortcut(),
                           "Low-overlap":  self.get_low_shortcut()}
        
        
        for do in ["High-overlap", "Low-overlap"]:
            for type in ["contradiction","entailment","neutral"]:

                type_selector = self.df_exp_set[do].gold_label == type 

                self.sets[do][type] = self.df_exp_set[do][type_selector].reset_index(drop=True)
                nums[do][type] = self.sets[do][type].shape[0]

        # get minimum size of samples
        self.type_balance = min({min(d.values()) for d in nums.values()})
        self.balance_sets = {}

        # Randomized Controlled Trials (RCTs)
        for do in ["High-overlap", "Low-overlap"]:
            
            self.balance_sets[do] = None
            frames = []

            # create an Empty DataFrame object
            for type in ["contradiction","entailment","neutral"]:
                
                # samples data
                ids = list(torch.randint(0, self.sets[do][type].shape[0], size=(self.type_balance,)))
                self.sets[do][type] = self.sets[do][type].loc[ids].reset_index(drop=True)

                #combine type 
                frames.append(self.sets[do][type])
            
            self.balance_sets[do] =  pd.concat(frames).reset_index(drop=True)

            assert self.balance_sets[do].shape[0] == (self.type_balance * 3)

            if self.is_group_by_class:
                
                self.premises[do] =  {}
                self.hypothesises[do] =  {}
                self.labels[do] = {}
                self.intervention[do] = {}
                
                for type in ["contradiction","entailment","neutral"]:

                    self.premises[do][type]  = list(self.sets[do][type].sentence1)
                    self.hypothesises[do][type]  = list(self.sets[do][type].sentence2)
                    self.labels[do][type]  = list(self.sets[do][type].gold_label)
                
                    self.intervention[do][type] = Intervention(encode = self.encode,
                                        premises = self.premises[do][type],
                                        hypothesises = self.hypothesises[do][type]
                                    )
            else:

                self.premises[do] = list(self.balance_sets[do].sentence1)
                self.hypothesises[do] = list(self.balance_sets[do].sentence2)
                self.labels[do] = list(self.balance_sets[do].gold_label)

                self.intervention[do] = Intervention(encode = self.encode,
                                        premises = self.premises[do],
                                        hypothesises = self.hypothesises[do]
                                    )

    def get_high_shortcut(self):

        # get high overlap score pairs
        return self.df[self.df['Treatment'] == "HOL"]

    def get_low_shortcut(self):

        # get low overlap score pairs
        return self.df[self.df['Treatment'] == "LOL"]
    
    def __len__(self):
        # Todo: generalize label

        if self.is_group_by_class:

            return self.type_balance

        else:
            return self.type_balance * len(set(self.labels['High-overlap']))
    
    def __getitem__(self, idx):

        pair_sentences = {}
        labels = {}

        if self.is_group_by_class:
            for do in ["High-overlap", "Low-overlap"]:
                
                pair_sentences[do] = {}
                labels[do] = {}

                for type in ["contradiction","entailment","neutral"]:

                    pair_sentences[do][type] = self.intervention[do][type].pair_sentences[idx]
                    labels[do][type] = self.labels[do][type][idx]
        else:

            for do in ["High-overlap", "Low-overlap"]:
                
                pair_sentences[do] = self.intervention[do].pair_sentences[idx]
                labels[do] = self.labels[do][idx]
        
        return pair_sentences , labels


class Dev(Dataset):
    def __init__(self, 
                data_path, 
                json_file, 
                DEBUG=False) -> None: 

        self.dev_name = list(json_file.keys())[0]
        data_path = os.path.join(data_path, json_file[self.dev_name])
        self.df = pd.read_json(data_path, lines=True)
        
        if self.dev_name == 'mismatched':
            if '-' in self.df.gold_label.unique(): 
                self.df = self.df[self.df.gold_label != '-'].reset_index(drop=True)


        self.premises = self.df.sentence1.tolist() if self.dev_name == "mismatched" else self.df.premise.tolist()
        self.hypos = self.df.sentence2.tolist() if self.dev_name == "mismatched" else self.df.hypothesis.tolist()
        self.labels = self.df.gold_label.tolist()

    def __len__(self):

        return self.df.shape[0]

    def __getitem__(self, idx):
        
        pair_sentence = [self.premises[idx], self.hypos[idx]]
        label = self.labels[idx] 
        
        return pair_sentence , label

def get_predictions(config, do, model, tokenizer, DEVICE, debug = False):

    low  =  config['params']['low'] #.7945 0.785  0.75   
    high =  config['params']['high']  #.7955 0.795  0.85
    step =  config['params']['step'] 
    digits = len(str(step).split('.')[-1])
    size= config['params']['size']
    mode = config['params']['mode']
    acc = {}

    layer = config['layer']

    torch.manual_seed(42)
    
    if config['intervention_type'] == "remove": epsilons = (low - high) * torch.rand(size) + high  # the interval (low, high)
    if config['intervention_type'] == "weaken": epsilons = [config['weaken']] if config['weaken'] is not None else [round(val, digits)for val in np.arange(low, high, step).tolist()]
    if config['intervention_type'] not in ["remove","weaken"]: epsilons = [0]
    if not isinstance(epsilons, list): epsilons = epsilons.tolist()

    epsilons = sorted(epsilons)
    mediators  = get_mediators(model)

    print(f"low :{low} , high : {high}")
    print(f"step : {step}")
    print(f"digits : {digits}")
    print(f"size : {size}")
    print(f"num epsilon : {len(epsilons)}")

    dev_set = Dev(config['dev_path'], config['dev_json'])
    dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)

    key = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
    top_k_mode =  'percent' if config['k'] is not None  else 'neurons' 
    
    # from validation(dev matched) set
    path = f'../pickles/top_neurons/top_neuron_{top_k_mode}_{do}_all_layers.pickle' if layer == -1 else f'../pickles/top_neurons/top_neuron_{top_k_mode}_{do}_{layer}.pickle'
    
    # get position of top neurons 
    with open(path, 'rb') as handle: top_neuron = pickle.load(handle) 
    
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

    cls = get_hidden_representations(config['counterfactual_paths'], 
                                    config['layers'], 
                                    config['heads'], 
                                    config['is_group_by_class'], 
                                    config['is_averaged_embeddings'])


    for epsilon in (t := tqdm(epsilons)): 
        

        prediction_path = '../pickles/prediction/' 
        
        prediction_path =  os.path.join(prediction_path, f'v{round(epsilon, digits)}')

        if not os.path.isdir(prediction_path): os.mkdir(prediction_path) 
        
        t.set_description(f"epsilon : {epsilon} , prediction path : {prediction_path}")
        
        for value in (n:= tqdm(num_neuron_groups)):
            
            if layer == -1:
                
                components = [neuron.split('-')[2] for neuron, v in top_neuron[value].items()]
                neuron_ids = [neuron.split('-')[3] for neuron, v in top_neuron[value].items()]
                layer_ids  = [neuron.split('-')[1] for neuron, v in top_neuron[value].items()]
                
            
            else:
                components = [neuron.split('-')[0] for neuron, v in top_neuron[value].items()]
                neuron_ids = [neuron.split('-')[1] for neuron, v in top_neuron[value].items()]
                layer_ids  =  [layer] * len(components)

            if config['single_neuron']: 
                
                layer_ids  =  [layer]
                components = [components[0]]
                neuron_ids = [neuron_ids[0]]
                
                raw_distribution_path = f'raw_distribution_{key}_{do}_L{layer}_{component}_{config["intervention_type"]}_{config["dev-name"]}.pickle'  

            else:
                
                if layer == -1:
                    raw_distribution_path = f'raw_distribution_{key}_{do}_all_layers_{value}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'  
                else:
                    raw_distribution_path = f'raw_distribution_{key}_{do}_L{layer}_{value}-k_{config["intervention_type"]}_{config["dev_name"]}.pickle'
                
            distributions = {}
            golden_answers = {}
            
            for mode in ["Null", "Intervene"]: 
                distributions[mode] = []
                golden_answers[mode] = []
            
            # test hans loader
            for batch_idx, (sentences, labels) in enumerate(dev_loader):

                premises, hypos = sentences

                pair_sentences = [[premise, hypo] for premise, hypo in zip(premises, hypos)]

                inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
                    
                inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

                #labels = [label_maps[label] for label in labels]
                # mediator used to intervene
                cur_dist = {}

                for mode in ["Null", "Intervene"]:

                    if mode == "Intervene": 

                        hooks = []
                        
                        for layer_id, component, neuron_id in zip(layer_ids, components, neuron_ids):

                            Z = cls[component][do][int(layer_id)]

                            hooks.append(mediators[component](int(layer_id)).register_forward_hook(neuron_intervention(
                                                                                        neuron_ids = [int(neuron_id)], 
                                                                                        component=component,
                                                                                        DEVICE = DEVICE ,
                                                                                        value = Z,
                                                                                        epsilon=epsilon,
                                                                                        intervention_type=config["intervention_type"],
                                                                                        debug=debug)))

                    with torch.no_grad(): 
                        
                        # Todo: generalize to distribution if the storage is enough
                        cur_dist[mode] = F.softmax(model(**inputs).logits , dim=-1)
                    
                    if mode == "Intervene": 
                        for hook in hooks: hook.remove() 

                    for sample_idx in range(cur_dist[mode].shape[0]):

                        distributions[mode].append(cur_dist[mode][sample_idx,:])
                        golden_answers[mode].append(labels[sample_idx]) 
            
            raw_distribution_path = os.path.join(prediction_path,  raw_distribution_path)
            
            with open(raw_distribution_path, 'wb') as handle: 
                pickle.dump(distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(golden_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving distributions and labels into : {raw_distribution_path}')

            
            if dev_set.dev_name != 'hans': acc[value] = compute_acc(raw_distribution_path, config["label_maps"])
                
        eval_path =  f'../pickles/evaluations/'
        eval_path =  os.path.join(eval_path, f'v{round(epsilon, digits)}')

        if not os.path.isdir(eval_path): os.mkdir(eval_path) 

        eval_path = os.path.join(eval_path, f'{key}_{do}_{config["intervention_type"]}_{config["dev_name"]}.pickle')
        
        with open(eval_path,'wb') as handle:
            pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saving all accuracies into {eval_path} ")
        
def prepare_result(raw_distribution_path, dev_set, component, do, layer, value, intervention_type, key, single_neuron=True):
    
    with open(raw_distribution_path, 'rb') as handle: 
        
        distributions = pickle.load(handle)
        golden_answers = pickle.load(handle)
        print(f'loading raw predictions from pickle: {raw_distribution_path}')

    text_answers = {}
    text_answer_path = None

    for mode in list(distributions.keys()):

        if mode not in text_answers.keys(): text_answers[mode] = []

        for sample_id in range(len(distributions[mode])):
        
            text_prediction = get_ans(torch.argmax(distributions[mode][sample_id], dim=-1))
            
            text_answers[mode].append(text_prediction)

    for mode in list(distributions.keys()):

        text_answer_path = f'../pickles/prediction/txt_answer_{mode}_{dev_set.dev_name}.txt'  
        
        # Todo: generalize to all challege sets
        if  os.path.exists(text_answer_path) and mode == 'Null': continue

        if mode == 'Intervene': 

            if single_neuron:

                text_answer_path = f'../pickles/prediction/txt_answer_{key}_{mode}_L{layer}_{component}_{do}_{intervention_type}_{dev_set.dev_name}.txt'  
            
            else:
            
                if layer == -1:
                    text_answer_path = f'../pickles/prediction/txt_answer_{key}_{mode}_all_layers_{value}-k_{do}_{intervention_type}_{dev_set.dev_name}.txt'  
                else:
                    text_answer_path = f'../pickles/prediction/txt_answer_{key}_{mode}_L{layer}_{value}-k_{do}_{intervention_type}_{dev_set.dev_name}.txt'  
        
        # Todo: write Null prediction if isn't exist
        
        # Todo: write Intervention prediction if isn't exist
        # in format : {mode}_L{layer}_{component}.txt
        # distributions[]
        
        with open(text_answer_path, "w") as fobj:

            headers = ['pairID', 'gold_label']

            fobj.write(f'{headers[0]}' + "," + f'{headers[1]}' +"\n")
            
            for sample_id, ans in enumerate(text_answers[mode]):

                fobj.write(f"ex{sample_id}" + "," + ans +"\n")

            print(f"saving text answer's bert predictions: {text_answer_path}")

def print_config(config):
            
    print(f"=========== Configs  ===============") 
    print(f"current experiment set :{config['exp_json']}")
    print(f"current dev set: {config['dev_json']}")
    print(f"is_group_by_class : {config['is_group_by_class']}")
    print(f"is_averaged_embeddings : {config['is_averaged_embeddings']}")
    print(f"+percent threshold of overlap score")
    print(f"upper_bound : {config['upper_bound']}")
    print(f"lower_bound : {config['lower_bound']}")
    print(f"samples used to compute nie scores : {config['num_samples']}") 
    print(f"Intervention type : {config['intervention_type']}")
    
    if config['k'] is not None : print(f"Top {config['k']} %k")
    if config['num_top_neurons'] is not None : print(f"Top number of neurons {config['num_top_neurons']}")

    if not config['getting_counterfactual']:

        print(f"HOL and LOL representation in the following paths ")

        for idx, path in enumerate(config['counterfactual_paths']):
            print(f"current: {path} ,  : {config['is_counterfactual_exist'][idx]} ")
    
    print(f"=========== End configs  =========") 

        
    


        

        





    


     


    