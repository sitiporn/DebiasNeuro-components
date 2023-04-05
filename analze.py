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
from utils import get_overlap_thresholds, group_by_treatment, test_mask, Classifier, get_hidden_representations
from utils import collect_output_components , report_gpu
from utils import geting_counterfactual_paths, get_single_representation, geting_NIE_paths
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

from data import ExperimentDataset
from intervention import intervene, high_level_intervention

class ComputingEmbeddings:
    def __init__(self, label_maps, label_remaps, tokenizer) -> None:
        
        self.representations = {}
        self.poolers = {}
        self.counter = {}
        
        self.acc = {}
        self.class_acc = {}
        self.confident = {}

        self.label_maps = label_maps
        self.label_remaps = label_remaps
        self.tokenizer = tokenizer

def cma_analysis(counterfactual_paths , save_nie_set_path, model, layers, treatments, heads, tokenizer, experiment_set, label_maps, is_averaged_embeddings , is_group_by_class,DEVICE, DEBUG=False):


    mediators = {}
    counter = None
    nie_dataset = None
    nie_dataloader = None
    counter_predictions  = {} 
    
    with open(save_nie_set_path, 'rb') as handle:
        
         nie_dataset = pickle.load(handle)
         nie_dataloader = pickle.load(handle)
        
         print(f"loading nie sets from pickle {save_nie_set_path} !")        
    
    # mediator used to intervene
    mediators["Q"] = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    mediators["K"] = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    mediators["V"] = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    mediators["AO"]  = lambda layer : model.bert.encoder.layer[layer].attention.output
    mediators["I"]  = lambda layer : model.bert.encoder.layer[layer].intermediate
    mediators["O"]  = lambda layer : model.bert.encoder.layer[layer].output

    if is_averaged_embeddings: 
        
        NIE = {}
        counter = {}
        
        cls = get_hidden_representations(counterfactual_paths, layers, heads, is_group_by_class, is_averaged_embeddings)

        high_level_intervention(nie_dataloader, mediators, cls, NIE, counter ,counter_predictions, layers, model, label_maps, tokenizer, treatments, DEVICE)
                
        NIE_path = f'../pickles/NIE/NIE_avg_high_level_{layers}_{treatments[0]}.pickle'
            
        with open(NIE_path, 'wb') as handle: 
            pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saving NIE scores into : {NIE_path}')

    else:
        
        for cur_path in (t := tqdm(counterfactual_paths)):
         
            counter = {}
            NIE = {}
            probs = {}

            # extract infor from current path 
            component = sorted(cur_path.split("_"), key=len)[0]  
            do = cur_path.split("_")[4]
            class_name = cur_path.split("_")[5]
            counterfactual_components = None
            
            t.set_description(f"Component : {component}")

            if do not in treatments and  component == "I": continue
             
            if component == "I":

                counterfactual_components = get_single_representation(cur_path, do = do, class_name = class_name)

                NIE_path = f'../pickles/individual_class_level/{layers}_{component}_{do}_{class_name}.pickle'
            
            else:

                counterfactual_components = get_single_representation(cur_path = cur_path)
                
                NIE_path = f'../pickles/individual_class_level/{layers}_{component}_{treatments[0]}.pickle'

            intervene(nie_dataloader, [component], mediators, counterfactual_components, NIE, counter, probs,counter_predictions, layers, model, label_maps, tokenizer, treatments, DEVICE)
            
            with open(NIE_path, 'wb') as handle: 
                
                pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving NIE scores into : {NIE_path}')

            
            del counterfactual_components
            report_gpu()
     
def get_top_k(NIE_paths, layers, treatments, k=5):

    topk = {"percent": (torch.tensor(list(range(1, k+1))) / 100).tolist()}
    top_neurons = {}
    num_neurons = None
        
    # compute average NIE
    for do in treatments:
        
        for layer in layers:
            
            ranking_nie = {}
        
            for cur_path in NIE_paths:
            
                with open(cur_path, 'rb') as handle:
                    
                    NIE = pickle.load(handle)
                    counter = pickle.load(handle)
                    
                    print(f"current : {cur_path}")

                for component in NIE[do].keys():
                
                    for neuron_id in NIE[do][component][layer].keys():
                        
                        NIE[do][component][layer][neuron_id] = NIE[do][component][layer][neuron_id] / counter[do][component][layer][neuron_id]

                        ranking_nie[component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
            
                    # Todo: get component and neuron_id and value 
            

            # top_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:5])
            all_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True))

            for percent in topk['percent']:
                
                num_neurons =  len(list(all_neurons.keys())) * percent
                num_neurons = int(num_neurons)

                print(f"++++++++ Component-Neuron_id: {round(percent, 2)} :+++++++++")
                
                top_neurons[round(percent, 2)] = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])

            with open(f'../pickles/top_neurons/top_neuron_{do}_{layer}.pickle', 'wb') as handle:
                pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Done saving top neurons into pickle !")

def compute_embedding_set(experiment_set, model, tokenizer, label_maps, DEVICE, is_group_by_class):
    
    label_remaps = { 0 :'contradiction', 1 : 'entailment', 2 : 'neutral'}

    computing_embeddings = ComputingEmbeddings(label_maps, label_remaps, tokenizer=tokenizer)

    classifier = Classifier(model=model)
        
    representation_loader = DataLoader(experiment_set,
                                        batch_size = 64,
                                        shuffle = False, 
                                        num_workers=0)
        
    for batch_idx, (sentences, labels) in enumerate(tqdm(representation_loader, desc=f"representation_loader")):
        
        for idx, do in enumerate(tqdm(['High-overlap','Low-overlap'], desc="Do-overlap")):
            
            if do not in computing_embeddings.representations.keys():

                if is_group_by_class:
                    
                    computing_embeddings.representations[do] = {}
                    computing_embeddings.poolers[do] = {}
                    computing_embeddings.counter[do] = {}


                    computing_embeddings.acc[do] = {}
                    computing_embeddings.class_acc[do] = {}
                    computing_embeddings.confident[do] = {}
                    
                    
                else:
                
                    computing_embeddings.representations[do] = []
                    computing_embeddings.poolers[do] = 0
                    computing_embeddings.counter[do] = 0
                
                    computing_embeddings.acc[do] = []
                    computing_embeddings.class_acc[do] = {"contradiction": [], "entailment" : [], "neutral" : []}
                    computing_embeddings.confident[do] = {"contradiction": 0, "entailment": 0, "neutral": 0}

            if experiment_set.is_group_by_class:

                for class_name in sentences[do].keys():

                    if class_name not in computing_embeddings.representations[do].keys():

                        computing_embeddings.representations[do][class_name] = []
                        computing_embeddings.poolers[do][class_name] = 0
                        computing_embeddings.counter[do][class_name] = 0


                    if class_name not in computing_embeddings.confident[do].keys():
                        
                        computing_embeddings.confident[do][class_name] =  0 #{"contradiction": 0, "entailment": 0, "neutral": 0}

                    if class_name not in computing_embeddings.class_acc[do].keys():

                        computing_embeddings.class_acc[do][class_name] =  [] #{"contradiction": [], "entailment" : [], "neutral" : []}

                        # each set divide into class level
                        #computing_embeddings.acc[do] = 0
                    

                    forward_pair_sentences(sentences[do][class_name],  computing_embeddings, labels[do][class_name], do, model, DEVICE, class_name)
            else:
                    
                forward_pair_sentences(sentences[do], computing_embeddings, labels[do], do, model, DEVICE)


    print(f"==== Distributions of Averaging representations across each set =====")
    
    # # Forward sentence to get distribution
    for do in ['High-overlap','Low-overlap']:
        
        print(f"++++++++++++++++++  {do} ++++++++++++++++++")

        if experiment_set.is_group_by_class:
                
            for class_name in ["contradiction", "entailment", "neutral"]:
            
                computing_embeddings.representations[do][class_name] = torch.stack(computing_embeddings.representations[do][class_name], dim=0)
                average_representation = torch.mean(computing_embeddings.representations[do][class_name], dim=0 ).unsqueeze(dim=0)

                out = classifier(average_representation).squeeze(dim=0)
                
                cur_distribution = F.softmax(out, dim=-1)

                # print(f"{class_name} set: {cur_distribution[label_maps[class_name]]}")
                print(f"{class_name} set: {cur_distribution}")

        else:

            computing_embeddings.representations[do] = torch.stack(computing_embeddings.representations[do], dim=0)
            average_representation = torch.mean(computing_embeddings.representations[do], dim=0 ).unsqueeze(dim=0)

            out = classifier(average_representation).squeeze(dim=0)
            
            cur_distribution = F.softmax(out, dim=-1)

            print(f"contradiction : {cur_distribution[label_maps['contradiction']]}")
            print(f"entailment : {cur_distribution[label_maps['entailment']]}")
            print(f"neutral : {cur_distribution[label_maps['neutral']]}")

        # print(f"contradiction : {cur_distribution[label_maps['contradiction']]}")
        # print(f"entailment : {cur_distribution[label_maps['entailment']]}")
        # print(f"neutral : {cur_distribution[label_maps['neutral']]}")

    print(f"====  Expected distribution of each set =====")

    for do in ['High-overlap','Low-overlap']:

        if is_group_by_class:
                
            #print(f"Overall accuray : {sum(computing_embeddings.acc[do]) / len(computing_embeddings.acc[do])}")
            print(f"++++++++++++++++++  {do} ++++++++++++++++++")
            
            for class_name in ["contradiction", "entailment", "neutral"]:

                computing_embeddings.confident[do][class_name] = computing_embeddings.confident[do][class_name].squeeze(dim=0)

                print(f"{class_name} set ; confident: {computing_embeddings.confident[do][class_name] / computing_embeddings.counter[do][class_name]}")

        else:
            
            print(f"++++++++++++++++++  {do} ++++++++++++++++++")
            print(f"Overall accuray : {sum(computing_embeddings.acc[do]) / len(computing_embeddings.acc[do])}")
            print(f"entail acc: {sum(computing_embeddings.class_acc[do]['entailment']) / len(computing_embeddings.class_acc[do]['entailment'])} ")   
            print(f"contradiction acc: {sum(computing_embeddings.class_acc[do]['contradiction']) / len(computing_embeddings.class_acc[do]['contradiction'])} ")
            print(f"neutral acc: {sum(computing_embeddings.class_acc[do]['neutral']) / len(computing_embeddings.class_acc[do]['neutral'])} ")
            
            print(f"******* expected distribution of golden answers ************")
            
            computing_embeddings.confident[do]['entailment'] = computing_embeddings.confident[do]['entailment'].squeeze(dim=0)
            computing_embeddings.confident[do]['contradiction'] = computing_embeddings.confident[do]['contradiction'].squeeze(dim=0) 
            computing_embeddings.confident[do]['neutral'] = computing_embeddings.confident[do]['neutral'].squeeze(dim=0)

            print(f"entail confident: {computing_embeddings.confident[do]['entailment'][computing_embeddings.label_maps['entailment']] / len(computing_embeddings.class_acc[do]['entailment'])} ")   
            print(f"contradiction confident: {computing_embeddings.confident[do]['contradiction'][computing_embeddings.label_maps['contradiction']] / len(computing_embeddings.class_acc[do]['contradiction'])} ")
            print(f"neutral confident: {computing_embeddings.confident[do]['neutral'][computing_embeddings.label_maps['neutral']]   / len(computing_embeddings.class_acc[do]['neutral'])}") 

    
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

def forward_pair_sentences(sentences, computing_embeddings, labels, do, model, DEVICE, class_name = None):
    
    LAST_HIDDEN_STATE = -1 
    CLS_TOKEN = 0

    classifier = Classifier(model=model)

    premise, hypo = sentences
    
    pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
    
    inputs = computing_embeddings.tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
    
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()} 

    golden_answers = [computing_embeddings.label_maps[label] for label in labels]
    golden_answers = torch.tensor(golden_answers).to(DEVICE)

    if class_name is None: 
        
        computing_embeddings.counter[do] += inputs['input_ids'].shape[0]
    else:
        
        computing_embeddings.counter[do][class_name] += inputs['input_ids'].shape[0]            
        
    with torch.no_grad(): 

        # Todo: generalize to distribution if the storage is enough
        outputs = model(**inputs, output_hidden_states=True)

        # (bz, seq_len, hiden_dim)
        representation = outputs.hidden_states[LAST_HIDDEN_STATE][:,CLS_TOKEN,:].unsqueeze(dim=1)

        predictions = torch.argmax(F.softmax(classifier(representation), dim=-1), dim=-1)

        # (bz, seq_len, hidden_dim)
        if class_name is None:
        
            # overall acc
            computing_embeddings.acc[do].extend((predictions == golden_answers).tolist())

            computing_embeddings.representations[do].extend(representation) 
            
             # by class
            for idx, label in enumerate(golden_answers.tolist()):

                computing_embeddings.confident[do][computing_embeddings.label_remaps[label]] += F.softmax(classifier(representation[idx,:,:].unsqueeze(dim=0)), dim=-1)
                computing_embeddings.class_acc[do][computing_embeddings.label_remaps[label]].extend([int(predictions[idx]) == label])

        else:

            # overall acc of current set
            # computing_embeddings.acc[do].extend((predictions == golden_answers).tolist())
            computing_embeddings.representations[do][class_name].extend(representation) 
             
             # by class
            for idx, label in enumerate(golden_answers.tolist()):

                # computing_embeddings.confident[do][class_name] += F.softmax(classifier(representation[idx,:,:].unsqueeze(dim=0)), dim=-1)
                # computing_embeddings.class_acc[do][class_name].extend([int(predictions[idx]) == label])
                
                computing_embeddings.confident[do][computing_embeddings.label_remaps[label]] += F.softmax(classifier(representation[idx,:,:].unsqueeze(dim=0)), dim=-1)
                computing_embeddings.class_acc[do][computing_embeddings.label_remaps[label]].extend([int(predictions[idx]) == label])
