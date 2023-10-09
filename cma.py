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
#from nn_pruning.patch_coordinator import (
#    SparseTrainingArguments,
#    ModelPatchingCoordinator,
#)


class ComputingEmbeddings:
    def __init__(self, label_maps, tokenizer) -> None:
        
        self.representations = {}
        self.poolers = {}
        self.counter = {}
        
        self.acc = {}
        self.class_acc = {}
        self.confident = {}

        self.label_maps = label_maps
        self.label_remaps = {v:k for k, v in self.label_maps.items()}
        self.tokenizer = tokenizer

def cma_analysis(config, model_path, seed, counterfactual_paths, NIE_paths, save_nie_set_path, model, treatments, tokenizer, experiment_set, DEVICE, DEBUG=False):
    # checking model and counterfactual_paths whether it change corresponding to seeds
    mediators = {}
    counter = None
    nie_dataset = None
    nie_dataloader = None
    counter_predictions  = {} 
    layers = config['layers']  if config['computed_all_layers'] else [config['layer']]
    assert len(layers) == 12, f"This doesn't cover all layers"
    NIE_path = { sorted(path.split('_'),key=len)[0]: path for path in NIE_paths} 
    NIE_path =  NIE_path['all'] if 'all' in NIE_path.keys() else NIE_path[str(config['layer'])]
    print(f"perform Causal Mediation analysis...")
    if model_path is not None: 
        _model = load_model(path= model_path, model=model)
    else:
        print(f'using original model as input to this function')
    with open(save_nie_set_path, 'rb') as handle:
        nie_dataset = pickle.load(handle)
        nie_dataloader = pickle.load(handle)
        print(f"loading nie sets from pickle {save_nie_set_path} !")        
    # mediator used to intervene corresponding to changing _model's seed
    mediators["Q"] = lambda layer : _model.bert.encoder.layer[layer].attention.self.query
    mediators["K"] = lambda layer : _model.bert.encoder.layer[layer].attention.self.key
    mediators["V"] = lambda layer : _model.bert.encoder.layer[layer].attention.self.value
    mediators["AO"]  = lambda layer : _model.bert.encoder.layer[layer].attention.output
    mediators["I"]  = lambda layer : _model.bert.encoder.layer[layer].intermediate
    mediators["O"]  = lambda layer : _model.bert.encoder.layer[layer].output

    if config['is_averaged_embeddings']: 
        NIE = {}
        counter = {}
        # Done checking counterfactual_paths change according to seed
        # Dont need model as input because we load counterfactual from -> counterfactual_paths
        # dont need head to specify components
        # cls shape: [seed][component][do][layer][neuron_ids]
        cls = get_hidden_representations(counterfactual_paths, layers, config['is_group_by_class'], config['is_averaged_embeddings'])
        # mediators:change respect to seed
        # cls: change respect to seed
        high_level_intervention(config, nie_dataloader, mediators, cls, NIE, counter , counter_predictions, layers, _model, config['label_maps'], tokenizer, treatments, DEVICE, seed=seed)
        
        with open(NIE_path, 'wb') as handle: 
            pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saving NIE scores into : {NIE_path}')
    else:
        for cur_path in (t := tqdm(config['counterfactual_paths'])):
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
            intervene(nie_dataloader, [component], mediators, counterfactual_components, NIE, counter, probs,counter_predictions, layers, _model, config['label_maps'], tokenizer, treatments, DEVICE)
            with open(NIE_path, 'wb') as handle: 
                pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving NIE scores into : {NIE_path}')
            del counterfactual_components
            report_gpu()
     
def get_topk(config, k=None, num_top_neurons=None):
    if config['eval_candidates']:
        topk = {'percent': k / 100}
    else: # ******************** Hyperparameter search ********************
        params  = get_params(config)
        total_neurons = get_num_neurons(config)
        if k is not None: topk = {"percent": (torch.tensor(list(range(1, k+1))) / 100).tolist()}
        if num_top_neurons is not None:
            topk = {"neurons": (torch.tensor(list(range(0, num_top_neurons+1, 5)))).tolist()} 
        else: 
            topk = {'percent': [config['masking_rate']] if config['masking_rate'] else params['percent']}
    return topk

def get_candidate_neurons(config, NIE_paths, treatments, debug=False):
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
    for cur_path in (t:=tqdm(NIE_paths)):
        # if ranking_nie is None: 
        ranking_nie = {}
        with open(cur_path, 'rb') as handle:
            NIE = pickle.load(handle)
            counter = pickle.load(handle)
            print(f"loading NIE : {cur_path}")
        # get seed number
        seed = cur_path.split('/')[2].split('_')[-1]
        # get treatment type
        do = cur_path.split('/')[-1].split('_')[2]
        t.set_description(f"{seed}, {do} : {cur_path}")
        # if seed not in ranking_nie.keys(): ranking_nie[seed] = {}
        for layer in layers:
            # layer = int(cur_path.split('_')[-2][1:-1])
            for component in NIE[do].keys():
                for neuron_id in NIE[do][component][layer].keys():
                    NIE[do][component][layer][neuron_id] = NIE[do][component][layer][neuron_id] / counter[do][component][layer][neuron_id]
                    # if config['computed_all_layers']:
                    #     ranking_nie[f"L-{layer}-"+ component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
                    # else:
                    #     ranking_nie[component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
                    ranking_nie[(f"L-{layer}-" if config['computed_all_layers'] else "") + component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
            # Todo: get component and neuron_id and value 
        # top_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:5])
        # sort layer each
        if not config['computed_all_layers']: 
            all_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True))
            for value in topk[key]:
                num_neurons =  len(list(all_neurons.keys())) * value if key == 'percent' else value
                num_neurons = int(num_neurons)
                print(f"++++++++ Component-Neuron_id: {round(value, 2) if key == 'percent' else num_neurons} neurons :+++++++++")
                top_neurons[round(value, 2) if key == 'percent' else num_neurons] = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])
            with open(f'../pickles/top_neurons/top_neuron_{seed}_{key}_{do}_{layer}.pickle', 'wb') as handle:
                pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Done saving top neurons into pickle !") 
        # sort whole layers
        if config['computed_all_layers']:
            all_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True))
            if not isinstance(topk[key], list): topk[key] = [topk[key]]
            for value in topk[key]:
                num_neurons =  len(list(all_neurons.keys())) * value if key == 'percent' else value
                num_neurons = int(num_neurons)
                print(f"++++++++ Component-Neuron_id: {round(value, 2) if key == 'percent' else num_neurons} neurons :+++++++++")
                top_neurons[round(value, 2) if key == 'percent' else value] = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])
            
            with open(f'../pickles/top_neurons/top_neuron_{seed}_{key}_{do}_all_layers.pickle', 'wb') as handle:
                pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Done saving top neurons into pickle !") 
            
            if debug:
                print(f"neurons:")
                print(list(top_neurons[0.01].keys())[:20])
                print(f"NIE values :")
                print(list(top_neurons[0.01].values())[:20])
        
        # with open(f'../pickles/top_neurons/top_neuron_{}_{key}_{do}_all_layers.pickle', 'rb') as handle:
        #     cur_top_neurons = pickle.load(handle)
        #     print(f"loading top neurons from pickles !") 


average_all_seed_distributions = {}
count_num_seed = 0
def evalutate_counterfactual(experiment_set, config, model, tokenizer, label_maps, DEVICE, is_group_by_class, seed=None, model_path=None, DEBUG=False, summarize=False):
    """ To see the difference between High-overlap and Low-overlap score whether our counterfactuals have huge different."""
    computing_embeddings = {}
    # To see the change of probs comparing between High bias and Low bias inputs
    global average_all_seed_distributions
    global count_num_seed 

    count_num_seed += 1

    computing_embeddings= ComputingEmbeddings(label_maps, tokenizer=tokenizer)
    
    if model_path is not None: 
        model = load_model(path= model_path, model=model)
    else:
        print(f'using original model as input to this function')
        
    classifier = Classifier(model=model)
    representation_loader = DataLoader(experiment_set, batch_size = 64, shuffle = False, num_workers=0)
        
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
            
            if do not in average_all_seed_distributions.keys():
                average_all_seed_distributions[do] = {"contradiction": 0, "entailment": 0, "neutral": 0}

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
    
    # **************** Compute for classifier output distributions given avg representation(High vs Low bias as input) to use as counterfactuals ****************
    if DEBUG: print(f"==== Classifier Output Distributions Given Averaging representations  as Input =====")
    for do in ['High-overlap','Low-overlap']:
        if DEBUG: print(f"++++++++++++++++++  {do} ++++++++++++++++++")
        if experiment_set.is_group_by_class:
            for class_name in ["contradiction", "entailment", "neutral"]:
                computing_embeddings.representations[do][class_name] = torch.stack(computing_embeddings.representations[do][class_name], dim=0)
                average_representation = torch.mean(computing_embeddings.representations[do][class_name], dim=0 ).unsqueeze(dim=0)
                out = classifier(average_representation).squeeze(dim=0)
                cur_distribution = F.softmax(out, dim=-1)
                # print(f"{class_name} set: {cur_distribution[label_maps[class_name]]}")
                if DEBUG: print(f"{class_name} set: {cur_distribution}")
        else:
            computing_embeddings.representations[do] = torch.stack(computing_embeddings.representations[do], dim=0)
            average_representation = torch.mean(computing_embeddings.representations[do], dim=0 ).unsqueeze(dim=0)
            # output the distribution using average representation as input to classififer 
            # rather than single representation
            out = classifier(average_representation).squeeze(dim=0)
            cur_distribution = F.softmax(out, dim=-1)

            for cur_class in label_maps.keys():
                if DEBUG: print(f"seed :{seed} {cur_class}: {cur_distribution[label_maps[cur_class]]}")
                average_all_seed_distributions[do][cur_class] += cur_distribution[label_maps[cur_class]]

    if DEBUG:
        for do in ['High-overlap','Low-overlap']:
            if is_group_by_class:
                print(f"++++++++++++++++++  {do} ++++++++++++++++++")
                for cur_class in label_maps.keys():
                    computing_embeddings.confident[do][class_name] = computing_embeddings.confident[do][class_name].squeeze(dim=0)
                    print(f"{class_name} set ; confident: {computing_embeddings.confident[do][class_name] / computing_embeddings.counter[do][class_name]}")
            else:
                print(f"++++++++++++++++++  {do} ++++++++++++++++++")
                print(f'Accuracies:')
                print(f"avg over all acc: {sum(computing_embeddings.acc[do]) / len(computing_embeddings.acc[do])}")
                for cur_class in label_maps.keys():
                    cur_score = sum(computing_embeddings.class_acc[do][cur_class]) / len(computing_embeddings.class_acc[do][cur_class])
                    print(f"{cur_class} acc: {cur_score} ")   
                
                print(f"Distributions:")
                for cur_class in label_maps.keys():
                    computing_embeddings.confident[do][cur_class] = computing_embeddings.confident[do][cur_class].squeeze(dim=0)
                    class_ind = computing_embeddings.label_maps[cur_class]
                    confident_score = computing_embeddings.confident[do][cur_class][class_ind] / len(computing_embeddings.class_acc[do][cur_class]) 
                    print(f"average {cur_class} confident: {confident_score}")
    
    if summarize and count_num_seed == 5:
        print('********** Counterfactual Model Output Distribution Summary **********')
        for do in ['High-overlap','Low-overlap']:
            print(f'>> {do}')
            for cur_class in label_maps.keys():
                print(f" {cur_class}: {average_all_seed_distributions[do][cur_class]/ count_num_seed}")

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
