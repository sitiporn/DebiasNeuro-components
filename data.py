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
from utils import get_ans


class ExperimentDataset(Dataset):
    def __init__(self, data_path, json_file, upper_bound, lower_bound, encode, is_group_by_class, num_samples, DEBUG=False) -> None: 
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


class Hans(Dataset):
    def __init__(self, 
                data_path, 
                json_file, 
                num_samples, 
                DEBUG=False) -> None: 

        data_path = os.path.join(data_path, json_file)

        self.df = pd.read_json(data_path, lines=True)

        self.premises = self.df.premise.tolist()
        self.hypos = self.df.hypothesis.tolist()
        self.labels = self.df.gold_label.tolist()

    def __len__(self):

        return self.df.shape[0]

    def __getitem__(self, idx):
        
        pair_sentence = [self.premises[idx], self.hypos[idx]]
        label = self.labels[idx] 
        
        return pair_sentence , label

def get_predictions(do,
                    layer,
                    model,
                    tokenizer,
                    DEVICE, 
                    layers, 
                    heads,
                    counterfactual_paths,
                    label_maps,
                    valid_path,
                    json_file,
                    is_group_by_class, 
                    is_averaged_embeddings,
                    intervention_type,
                    single_neuron = False):

    mediators  = get_mediators(model)

    hans_set = Hans(valid_path,
                    json_file,
                    num_samples=300)

    hans_loader = DataLoader(hans_set, 
                        batch_size = 32,
                        shuffle = False, 
                        num_workers=0)

    
    path = f'../pickles/top_neurons/top_neuron_{do}_{layer}.pickle'
    prediction_path = None
    
    with open(path, 'rb') as handle:
        # get [CLS] activation 
        top_neuron = pickle.load(handle)

    cls = get_hidden_representations(counterfactual_paths, 
                                    layers, 
                                    heads, 
                                    is_group_by_class, 
                                    is_averaged_embeddings)
    

    components = [neuron.split('-')[0] for neuron, v in top_neuron.items()]
    neuron_ids = [neuron.split('-')[1] for neuron, v in top_neuron.items()]

    if single_neuron: 
        components = [components[0]]
        neuron_ids = [neuron_ids[0]]
        prediction_path = f'../pickles/prediction/{do}_L{layer}_{component}.pickle'  
    else:
        prediction_path = f'../pickles/prediction/{do}_L{layer}_top-k.pickle'  

    print(f"++++  All mediators ++++")
    print(f"components : {components}") 
    print(f"neurons : {neuron_ids}") 

    distributions = {}
    golden_answers = {}
    
    for mode in ["Null", "Intervene"]: 
        distributions[mode] = []
        golden_answers[mode] = []
    
    # test hans loader
    for batch_idx, (sentences, labels) in enumerate(hans_loader):

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
                
                for component, neuron_id in zip(components, neuron_ids):

                    Z = cls[component][do][layer]

                    hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(
                                                                                neuron_ids = [int(neuron_id)], 
                                                                                component=component,
                                                                                DEVICE = DEVICE ,
                                                                                value = Z,
                                                                                intervention_type=intervention_type)))

            with torch.no_grad(): 
                
                # Todo: generalize to distribution if the storage is enough
                cur_dist[mode] = F.softmax(model(**inputs).logits , dim=-1)
            
            if mode == "Intervene": 
                for hook in hooks: hook.remove() 

            for sample_idx in range(cur_dist[mode].shape[0]):

                distributions[mode].append(cur_dist[mode][sample_idx,:])
                golden_answers[mode].append(labels[sample_idx])


    
    with open(prediction_path, 'wb') as handle: 
        
        pickle.dump(distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(golden_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving distributions and labels into : {prediction_path}')

    prepare_result(prediction_path=prediction_path, 
                  hans_set=hans_set,
                  component=component,
                  do=do,
                  layer=layer,
                  single_neuron=single_neuron)

def prepare_result(prediction_path, hans_set, component, do, layer, single_neuron=True):
    
    with open(prediction_path, 'rb') as handle: 
        
        distributions = pickle.load(handle)
        golden_answers = pickle.load(handle)
        print(f'loading predictions from : {prediction_path}')

    text_answers = {}
    txt_path = None

    for mode in list(distributions.keys()):

        if mode not in text_answers.keys(): text_answers[mode] = []

        for sample_id in range(len(distributions[mode])):
        
            text_prediction = get_ans(torch.argmax(distributions[mode][sample_id], dim=-1))
            
            text_answers[mode].append(text_prediction)


    # Todo: create text file for Null and Intervention


    # headers = ['pairID', 'gold_label']
    # file = open('filepath','a')

    for mode in list(distributions.keys()):

        txt_path  = f'../pickles/prediction/bert_{mode}.txt'  
        
        # Todo: generalize to all challege sets
        if  os.path.exists(txt_path) and mode == 'Null': continue

        if mode == 'Intervene': 

            if single_neuron:
                 txt_path = f'../pickles/prediction/bert_{mode}_L{layer}_{component}_{do}.txt'  
            else:
                 txt_path = f'../pickles/prediction/bert_{mode}_L{layer}_top-k_{do}.txt'  
        
        # Todo: write Null prediction if isn't exist

        
        # Todo: write Intervention prediction if isn't exist
        # in format : {mode}_L{layer}_{component}.txt
        # distributions[]
        
        with open(txt_path, "w") as fobj:

            headers = ['pairID', 'gold_label']

            fobj.write(f'{headers[0]}' + "," + f'{headers[1]}' +"\n")
            
            for sample_id, ans in enumerate(text_answers[mode]):

                fobj.write(f"ex{sample_id}" + "," + ans +"\n")
        
    
    


        

        





    


     


    