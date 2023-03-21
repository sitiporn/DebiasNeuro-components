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

def neuron_intervention(neuron_ids, 
                       DEVICE,
                       value,
                       intervention_type='replace'):
    
    # Hook for changing representation during forward pass
    def intervention_hook(module,
                            input,
                            output):
        
        # define mask where to overwrite
        scatter_mask = torch.zeros_like(output, dtype = torch.bool)

        # where to intervene
        # bz, seq_len, hidden_dim
        scatter_mask[:,0, neuron_ids] = 1
        
        neuron_values = value[neuron_ids]

        neuron_values = neuron_values.repeat(output.shape[0], output.shape[1], 1).to(DEVICE)
        
        output.masked_scatter_(scatter_mask, neuron_values)

    return intervention_hook


# ************  intervention  *******************
def high_level_intervention(nie_dataloader, mediators, cls, NIE, counter ,counter_predictions, layers, model, label_maps, tokenizer, treatments, DEVICE):

    components = cls.keys()

    for batch_idx, (sentences, labels) in (t := tqdm(enumerate(nie_dataloader))):
        
        t.set_description(f"NIE_dataloader, batch_idx : {batch_idx}")

        premise, hypo = sentences

        pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]
            
        # distributions 
        probs = {}
            
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
            
        with torch.no_grad(): 
            
            # Todo: generalize to distribution if the storage is enough
            probs['null'] = F.softmax(model(**inputs).logits , dim=-1)[:, label_maps["entailment"]]
            
        # To store all positions
        probs['intervene'] = {}
        
        # run one full neuron intervention experiment
        for do in treatments: 
            
            if do not in NIE.keys():
                NIE[do] = {}
                counter[do]  = {} 
                probs['intervene'][do] = {}

            for component in components: 

                if  component not in NIE[do].keys():
                   
                    NIE[do][component] = {}
                    counter[do][component] = {}
                
                for layer in layers:
                    
                    if  layer not in NIE[do][component].keys(): 

                        NIE[do][component][layer] = {}
                        counter[do][component][layer] = {}
                        
                    Z = cls[component][do][layer]
                
                    for neuron_id in range(Z.shape[0]):
                    
                        hooks = [] 
                        
                        hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(neuron_ids = [neuron_id], DEVICE = DEVICE ,value = Z)))
                        
                        with torch.no_grad(): 

                            intervene_probs = F.softmax(model(**inputs).logits , dim=-1)

                            entail_probs = intervene_probs[:, label_maps["entailment"]]
                            
                        if neuron_id not in NIE[do][component][layer].keys():
                            NIE[do][component][layer][neuron_id] = 0
                            counter[do][component][layer][neuron_id] = 0

                        
                        NIE[do][component][layer][neuron_id] += torch.sum( (entail_probs / probs['null'])-1, dim=0)
                        counter[do][component][layer][neuron_id] += entail_probs.shape[0]

                        for hook in hooks: hook.remove() 

def intervene(dataloader, components, mediators, cls, NIE, counter, probs, counter_predictions, layers, model, label_maps, tokenizer, treatments, DEVICE):
    
    # Todo: change  dataloader to w/o group by class
    for nie_class_name in dataloader.keys():
      
        print(f"NIE class: {nie_class_name}")
        
        for batch_idx, (sentences, labels) in (d := tqdm(enumerate(dataloader[nie_class_name]))):

            d.set_description(f"NIE_dataloader, batch_idx : {batch_idx}")

            if batch_idx == 2:
                break

            premise, hypo = sentences

            pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]

            
            inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
            
            inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

            with torch.no_grad(): 
                
                # Todo: generalize to distribution if the storage is enough
                null_probs = F.softmax(model(**inputs).logits , dim=-1)

            # To store all positions
            probs['intervene'] = {}

            # run one full neuron intervention experiment
            for do in treatments: 
                
                if do not in NIE.keys():
                    NIE[do] = {}
                    counter[do]  = {} 
                    probs['intervene'][do] = {}

                for component in components: 

                    if  component not in NIE[do].keys():
                        NIE[do][component] = {}
                        counter[do][component]  = {} 
                        probs['intervene'][do][component] = {}
                    
                    for layer in layers:
                        
                        if  layer not in NIE[do][component].keys(): 

                            NIE[do][component][layer] = {}
                            counter[do][component][layer]  = {} 
                            probs['intervene'][do][component][layer]  = {} 

                        for counterfactual_class_name in cls[component][do].keys(): 
                            
                            # t_counterfactual_class.set_description(f"NIE class: {counterfactual_class_name}")

                            if counterfactual_class_name != nie_class_name: continue
                            
                            for counterfactual_idx in range(len(cls[component][do][counterfactual_class_name][layer])):

                                Z = cls[component][do][counterfactual_class_name][layer][counterfactual_idx]
                            
                                for neuron_id in range(Z.shape[0]):
                                        
                                    hooks = [] 
                                    
                                    hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(neuron_ids = [neuron_id], DEVICE = DEVICE ,value = Z)))
                                    
                                    with torch.no_grad(): 

                                        intervene_probs = F.softmax(model(**inputs).logits , dim=-1)

                                    if neuron_id not in NIE[do][component][layer].keys():
                                        NIE[do][component][layer][neuron_id] = 0
                                        counter[do][component][layer][neuron_id]  = 0 
                                        probs['intervene'][do][component][layer][neuron_id]  = []

                                    ret = (intervene_probs[:, label_maps["entailment"]] / null_probs[:, label_maps["entailment"]]) 
                                    
                                    NIE[do][component][layer][neuron_id] += torch.sum(ret - 1, dim=0)
                                    counter[do][component][layer][neuron_id] += intervene_probs.shape[0]
                                    probs['intervene'][do][component][layer][neuron_id].append(intervene_probs)
                                    
                                    for hook in hooks: hook.remove() 
                                    
def prunning(model, layers):

    # Todo: generalize for any models
    # ref- https://arxiv.org/pdf/2210.16079.pdf
    # Todo: get Wl Wl_K , Wl_Q, Wl_V , Wl_AO, Wl_I , Wl_O of layer
    Wl_Q = lambda layer : model.bert.encoder.layer[layer].attention.self.query.weight.data
    Wl_K = lambda layer : model.bert.encoder.layer[layer].attention.self.key.weight.data
    Wl_V = lambda layer : model.bert.encoder.layer[layer].attention.self.value.weight.data
    Wl_AO = lambda layer : model.bert.encoder.layer[layer].output.dense.weight.data
    Wl_I  = lambda layer : model.bert.encoder.layer[layer].intermediate.dense.weight.data
    Wl_O =  lambda layer : model.bert.encoder.layer[layer].output.dense.weight.data

    for layer in layers:
        # inital all mask are value
        Ml_Q = torch.zeros_like(Wl_Q(layer))
        Ml_K = torch.zeros_like(Wl_K(layer))
        Ml_V = torch.zeros_like(Wl_V(layer))
        Ml_AO = torch.zeros_like(Wl_AO(layer))
        Ml_I  = torch.zeros_like(Wl_I(layer))
        Ml_O = torch.zeros_like(Wl_O(layer))

        # Todo: change to data.copy mode
        with torch.no_grad(): 
            model.bert.encoder.layer[layer].attention.self.query.weight.data.copy_(Wl_Q(layer) *  Ml_Q )
            # model.bert.encoder.layer[layer].attention.self.key.weight = Wl_K(layer) *  Ml_K 
            # model.bert.encoder.layer[layer].attention.self.key.value = Wl_V(layer) *  Ml_V 
            # model.bert.encoder.layer[layer].output.dense.weight = Wl_AO(layer) *  Ml_AO
            # model.bert.encoder.layer[layer].intermediate.dense = Wl_I(layer) *  Ml_I 
            # model.bert.encoder.layer[layer].output.dense = Wl_O(layer) *  Ml_O 

class Intervention():
    """Wrapper all possible interventions """
    def __init__(self, 
                encode, 
                premises: list, 
                hypothesises: list, 
                device = 'cpu') -> None:

        super()
        
        self.encode = encode
        self.premises = premises
        self.hypothesises = hypothesises
        
        self.pair_sentences = []
        
        for premise, hypothesis in zip(self.premises, self.hypothesises):

            # Encode a pair of sentences and make a prediction
            self.pair_sentences.append([premise, hypothesis])

        # Todo : sort text before encode to reduce  matrix size of each batch
        # self.batch_tok = collate_tokens([self.encode(pair[0], pair[1]) for pair in self.batch_of_pairs], pad_idx=1)
        # self.batch_tok = self.encode(self.premises, self.hypothesises,truncation=True, padding="max_length")

        """
        # All the initial strings
        # First item should be neutral, others tainted ? 
        
        self.base_strings = [base_string.format(s) for s in substitutes]

        # Where to intervene
        # Text position ?
        self.position = base_string.split().index('{}')
        """