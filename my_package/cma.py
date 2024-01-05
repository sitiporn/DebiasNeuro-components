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
    # mediator used to intervene corresponding to changing _model's seed
    mediators["Q"]  = lambda layer : _model.bert.encoder.layer[layer].attention.self.query
    mediators["K"]  = lambda layer : _model.bert.encoder.layer[layer].attention.self.key
    mediators["V"]  = lambda layer : _model.bert.encoder.layer[layer].attention.self.value
    mediators["AO"] = lambda layer : _model.bert.encoder.layer[layer].attention.output
    mediators["I"]  = lambda layer : _model.bert.encoder.layer[layer].intermediate
    mediators["O"]  = lambda layer : _model.bert.encoder.layer[layer].output

    if config['is_averaged_embeddings']: 
        NIE = {}
        counter = {}
        # cls shape: [seed][component][do][layer][neuron_ids]
        cls = get_hidden_representations(config, counterfactual_paths, method_name, seed,layers, config['is_group_by_class'], config['is_averaged_embeddings'])
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
    mode = config['top_neuron_mode']
    print(f'get_candidate_neurons: {mode}')
    # select candidates based on the number of neurons
    num_top_neurons = config['num_top_neurons']
    top_neuron_path = f'../pickles/top_neurons/{method_name}/'
    if not os.path.exists(top_neuron_path): os.mkdir(top_neuron_path)
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
        seed = cur_path.split('/')[3].split('_')[-1]
        # get treatment type
        do = cur_path.split('/')[-1].split('_')[2]
        t.set_description(f"{seed}, {do} : {cur_path}")
        if seed is None: seed = str(seed)
        for layer in layers:
            # layer = int(cur_path.split('_')[-2][1:-1])
            for component in NIE[do].keys():
                for neuron_id in NIE[do][component][layer].keys():
                    NIE[do][component][layer][neuron_id] = NIE[do][component][layer][neuron_id] / counter[do][component][layer][neuron_id]
                    ranking_nie[(f"L-{layer}-" if config['computed_all_layers'] else "") + component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
        # sort layer each
        if not config['computed_all_layers']: 
            all_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True))
            for value in topk[key]:
                num_neurons =  len(list(all_neurons.keys())) * value if key == 'percent' else value
                num_neurons = int(num_neurons) 
                print(f"++++++++ Component-Neuron_id: {round(value, 4) if key == 'percent' else num_neurons} neurons :+++++++++")
                top_neurons[round(value, 4) if key == 'percent' else num_neurons] = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])
            with open(os.path.join(top_neuron_path, f'top_neuron_{seed}_{key}_{do}_{layer}.pickle'), 'wb') as handle:
                pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Done saving top neurons into pickle !") 
        # sort whole layers
        if config['computed_all_layers']:
            all_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True))
            if not isinstance(topk[key], list): topk[key] = [topk[key]]
            for value in topk[key]:
                num_neurons =  len(list(all_neurons.keys())) * value if key == 'percent' else value
                num_neurons = int(num_neurons)
                print(f"++++++++ Component-Neuron_id: {round(value, 4) if key == 'percent' else num_neurons} neurons :+++++++++")
                
                if mode == 'random':
                    from operator import itemgetter
                    cur_neurons = sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)
                    random.shuffle(cur_neurons)
                    top_neurons[round(value, 4) if key == 'percent' else value] = dict(cur_neurons[:num_neurons])
                    # ids = []
                    # while len(set(ids)) < num_neurons:
                    #     id = int(torch.randint(num_neurons, len(cur_neurons), size=(1,)))
                    #     ids.append(id)
                    #     ids = list(set(ids))
                    # assert len(ids) == len(set(ids)), f"len {len(ids)}, set len: {len(set(ids))}"
                    # assert len(ids) == num_neurons
                    # top_neurons[round(value, 4) if key == 'percent' else value] = dict(itemgetter(*ids)(cur_neurons))
                elif mode == 'sorted':
                    top_neurons[round(value, 4) if key == 'percent' else value] = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:num_neurons])
            
            if mode == 'random':
                save_path = os.path.join(top_neuron_path, f'random_top_neuron_{seed}_{key}_{do}_all_layers.pickle')
                with open(save_path, 'wb') as handle:
                    pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Done saving random top neurons into pickle! : {save_path}") 
            
            elif mode == 'sorted':
                save_path = os.path.join(top_neuron_path, f'top_neuron_{seed}_{key}_{do}_all_layers.pickle')
                with open(save_path, 'wb') as handle:
                    pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Done saving top neurons into pickle!: {save_path}") 
             
            if debug:
                print(f"neurons:")
                print(list(top_neurons[0.01].keys())[:20])
                print(f"NIE values :")
                print(list(top_neurons[0.01].values())[:20])


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
        treatments = ['High-overlap'] if config['dataset_name'] == 'fever' else ['High-overlap','Low-overlap']
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

def forward_pair_sentences(sentences, counterfactuals, labels, do, model, DEVICE, class_name = None):
    
    LAST_HIDDEN_STATE = -1 
    CLS_TOKEN = 0
    classifier = Classifier(model=model)
    
    premise, hypo = sentences
    pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
    inputs = counterfactuals.tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k,v in inputs.items()} 

    golden_answers = [counterfactuals.label_maps[label] for label in labels]
    golden_answers = torch.tensor(golden_answers).to(DEVICE)

    if class_name is None: 
        counterfactuals.counter[do] += inputs['input_ids'].shape[0]
    else:
        counterfactuals.counter[do][class_name] += inputs['input_ids'].shape[0]            

    breakpoint() 
    with torch.no_grad(): 
        # Todo: generalize to distribution if the storage is enough
        outputs = model(**inputs, output_hidden_states=True)
        # (bz, seq_len, hiden_dim)
        representation = outputs.hidden_states[LAST_HIDDEN_STATE][:,CLS_TOKEN,:].unsqueeze(dim=1)
        predictions = torch.argmax(F.softmax(classifier(representation), dim=-1), dim=-1)

        # (bz, seq_len, hidden_dim)
        if class_name is None:
            # overall acc
            counterfactuals.acc[do].extend((predictions == golden_answers).tolist())
            counterfactuals.representations[do].extend(representation) 
             # by class
            for idx, label in enumerate(golden_answers.tolist()):
                counterfactuals.confident[do][counterfactuals.label_remaps[label]] += F.softmax(classifier(representation[idx,:,:].unsqueeze(dim=0)), dim=-1)
                counterfactuals.class_acc[do][counterfactuals.label_remaps[label]].extend([int(predictions[idx]) == label])
        else:
            # overall acc of current set
            # counterfactuals.acc[do].extend((predictions == golden_answers).tolist())
            counterfactuals.representations[do][class_name].extend(representation) 
             # by class
            for idx, label in enumerate(golden_answers.tolist()):
                # counterfactuals.confident[do][class_name] += F.softmax(classifier(representation[idx,:,:].unsqueeze(dim=0)), dim=-1)
                # counterfactuals.class_acc[do][class_name].extend([int(predictions[idx]) == label])
                counterfactuals.confident[do][counterfactuals.label_remaps[label]] += F.softmax(classifier(representation[idx,:,:].unsqueeze(dim=0)), dim=-1)
                counterfactuals.class_acc[do][counterfactuals.label_remaps[label]].extend([int(predictions[idx]) == label])

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