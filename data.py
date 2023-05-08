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
from utils import get_num_neurons, get_params


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

def get_predictions(config, do,  model, tokenizer, DEVICE, debug = False):

    acc = {}

    layer = config['layer']

    torch.manual_seed(42)

    params, digits = get_params(config)
    total_neurons = get_num_neurons(config)

    epsilons = params['epsilons']
    
    if not isinstance(epsilons, list): epsilons = epsilons.tolist()

    epsilons = sorted(epsilons)
    mediators  = get_mediators(model)

    dev_set = Dev(config['dev_path'], config['dev_json'])
    dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)

    key = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
    top_k_mode =  'percent' if config['k'] is not None  else 'neurons' 
    
    # from validation(dev matched) set
    path = f'../pickles/top_neurons/top_neuron_{top_k_mode}_{do}_all_layers.pickle' if layer == -1 else f'../pickles/top_neurons/top_neuron_{top_k_mode}_{do}_{layer}.pickle'
    
    # get position of top neurons 
    with open(path, 'rb') as handle: top_neuron = pickle.load(handle) 
    
    # Todo: changing neuron group correspond to percent
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

    cls = get_hidden_representations(config['counterfactual_paths'], 
                                    config['layers'], 
                                    config['heads'], 
                                    config['is_group_by_class'], 
                                    config['is_averaged_embeddings'])


    for epsilon in (t := tqdm(epsilons)): 
        

        prediction_path = '../pickles/prediction/' 
        
        prediction_path =  os.path.join(prediction_path, f'v{round(epsilon, digits["epsilons"])}')

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
                    raw_distribution_path = f'raw_distribution_{key}_{do}_L{layer}_{value}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'
                
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

        eval_path = os.path.join(eval_path, f'{key}_{do}_{config["intervention_type"]}_{config["dev-name"]}.pickle')
        
        with open(eval_path,'wb') as handle:
            pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saving all accuracies into {eval_path} ")
        
def convert_to_text_ans(config, neuron_path, params, digits):

    """ changing distributions into text anaswers on hans set"""

    with open(neuron_path, 'rb') as handle: 
        top_neuron = pickle.load(handle)

    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

    low  =  config['percent']['low'] 
    high =  config['percent']['high']  
    step =  config['percent']['step'] 

    digits = len(str(step).split('.')[-1])
    
    size = config['percent']['size']
    mode = config['percent']['mode']
    
    layer = config['layer']
    rank_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
    prediction_path = '../pickles/prediction/' 

    if config['intervention_type'] == "remove": epsilons = (low - high) * torch.rand(size) + high  # the interval (low, high)
    if config['intervention_type'] == "weaken": epsilons = [config['weaken']] if config['weaken'] is not None else [round(val, digits)for val in np.arange(low, high, step).tolist()]
    if config['intervention_type'] not in ["remove","weaken"]: epsilons = [0]

    for epsilon in (t := tqdm(epsilons)):  

        # read pickle file used to interpret as text answers later
        epsilon_path = f'v{round(epsilon, digits)}'
       
        for neurons in (n:= tqdm(num_neuron_groups)):

            # why top neuron doesnt show result
            # key : (percent, neuron, weaken)
            # value : neuron_group
        
            # dont touch this 
            raw_distribution_path = f'raw_distribution_{rank_mode}_{config["eval"]["do"]}_all_layers_{neurons}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'  
            raw_distribution_path = os.path.join(os.path.join(prediction_path, epsilon_path),  raw_distribution_path)
            
            # bugs: no such file because no intervention yet
            with open(raw_distribution_path, 'rb') as handle: 
                distributions = pickle.load(handle)
                golden_answers = pickle.load(handle)
            
            text_answers = {}
            text_answer_path = None

            for mode in list(distributions.keys()):

                if mode not in text_answers.keys(): text_answers[mode] = []

                for sample_id in range(len(distributions[mode])):
                
                    text_prediction = get_ans(torch.argmax(distributions[mode][sample_id], dim=-1))
                    
                    text_answers[mode].append(text_prediction)

            for mode in list(distributions.keys()):

                text_answer_path = f'txt_answer_{mode}_{config["dev-name"]}.txt'  
                
                # Todo: generalize to all challege sets
                if  os.path.exists(text_answer_path) and mode == 'Null': continue

                if mode == 'Intervene': 

                    if config["single_neuron"]:

                        component = [neuron.split('-')[2 if layer == -1 else 0] for neuron, v in top_neuron[neurons].items()][0]

                        text_answer_path = f'txt_answer_{rank_mode}_{mode}_L{layer}_{component}_{config["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
                    
                    else:
                    
                        if layer == -1:
                            text_answer_path = f'txt_answer_{rank_mode}_{mode}_all_layers_{neurons}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
                        else:
                            text_answer_path = f'txt_answer_{rank_mode}_{mode}_L{layer}_{neurons}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
            
            text_answer_path  = os.path.join(os.path.join(prediction_path, epsilon_path), text_answer_path)

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

def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"

def get_result(config, params, eval_path, prediction_path, neuron_path, top_neuron, digits, prediction_mode):
    
    if config['to_text']: convert_to_text_ans(config, neuron_path, params, digits)
    
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

    for epsilon in (t := tqdm(params['epsilons'])):  

        epsilon_path = f'v{round(epsilon, digits["epsilons"])}'

        t.set_description(f"epsilon : {epsilon} ")
        
        for group in num_neuron_groups:

            text_answer_path = f'txt_answer_{prediction_mode}_{config["eval"]["intervention_mode"]}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
            result_path = f'result_{prediction_mode}_{config["eval"]["intervention_mode"]}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

            if config['eval']['all_layers']: text_answer_path = f'txt_answer_{prediction_mode}_{config["eval"]["intervention_mode"]}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
            if config['eval']['all_layers']: result_path = f'result_{prediction_mode}_{config["eval"]["intervention_mode"]}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

            config['evaluations'][group] = {}

            text_answer_path = os.path.join(os.path.join(prediction_path, epsilon_path),  text_answer_path)
            result_path = os.path.join(os.path.join(eval_path, epsilon_path),  result_path)

            tables = {}

            fi = open(text_answer_path, "r")

            first = True
            guess_dict = {}

            for line in fi:
                if first:
                    first = False
                    continue
                else:
                    parts = line.strip().split(",")
                    guess_dict[parts[0]] = format_label(parts[1])

            # load from hans set up
            fi = open("../hans/heuristics_evaluation_set.txt", "r")

            correct_dict = {}
            first = True

            heuristic_list = []
            subcase_list = []
            template_list = []

            for line in fi:
                if first:
                    labels = line.strip().split("\t")
                    idIndex = labels.index("pairID")
                    first = False
                    continue
                else:
                    parts = line.strip().split("\t")
                    this_line_dict = {}
                    for index, label in enumerate(labels):
                        if label == "pairID":
                            continue
                        else:
                            this_line_dict[label] = parts[index]
                    correct_dict[parts[idIndex]] = this_line_dict

                    if this_line_dict["heuristic"] not in heuristic_list:
                        heuristic_list.append(this_line_dict["heuristic"])
                    if this_line_dict["subcase"] not in subcase_list:
                        subcase_list.append(this_line_dict["subcase"])
                    if this_line_dict["template"] not in template_list:
                        template_list.append(this_line_dict["template"])

            heuristic_ent_correct_count_dict = {}
            subcase_correct_count_dict = {}
            template_correct_count_dict = {}
            heuristic_ent_incorrect_count_dict = {}
            subcase_incorrect_count_dict = {}
            template_incorrect_count_dict = {}
            heuristic_nonent_correct_count_dict = {}
            heuristic_nonent_incorrect_count_dict = {}

            for heuristic in heuristic_list:
                heuristic_ent_correct_count_dict[heuristic] = 0
                heuristic_ent_incorrect_count_dict[heuristic] = 0
                heuristic_nonent_correct_count_dict[heuristic] = 0 
                heuristic_nonent_incorrect_count_dict[heuristic] = 0

            for subcase in subcase_list:
                subcase_correct_count_dict[subcase] = 0
                subcase_incorrect_count_dict[subcase] = 0

            for template in template_list:
                template_correct_count_dict[template] = 0
                template_incorrect_count_dict[template] = 0

            for key in correct_dict:
                traits = correct_dict[key]
                heur = traits["heuristic"]
                subcase = traits["subcase"]
                template = traits["template"]

                guess = guess_dict[key]
                correct = traits["gold_label"]

                if guess == correct:
                    if correct == "entailment":
                        heuristic_ent_correct_count_dict[heur] += 1
                    else:
                        heuristic_nonent_correct_count_dict[heur] += 1

                    subcase_correct_count_dict[subcase] += 1
                    template_correct_count_dict[template] += 1
                else:
                    if correct == "entailment":
                        heuristic_ent_incorrect_count_dict[heur] += 1
                    else:
                        heuristic_nonent_incorrect_count_dict[heur] += 1
                    subcase_incorrect_count_dict[subcase] += 1
                    template_incorrect_count_dict[template] += 1

            tables['correct']  = { 'entailed': heuristic_ent_correct_count_dict, 'non-entailed': heuristic_nonent_correct_count_dict}
            tables['incorrect'] = { 'entailed': heuristic_ent_incorrect_count_dict,  'non-entailed': heuristic_nonent_incorrect_count_dict}

            for cur_class in ['entailed','non-entailed']:

                print(f"Heuristic  {cur_class} results:")

                if cur_class not in config["evaluations"][group].keys():  config["evaluations"][group][cur_class] = {}

                for heuristic in heuristic_list:

                    correct = tables['correct'][cur_class][heuristic]
                    incorrect = tables['incorrect'][cur_class][heuristic]

                    total = correct + incorrect
                    percent = correct * 1.0 / total
                    print(heuristic + ": " + str(percent))

                    config["evaluations"][group][cur_class][heuristic] = percent

            
            with open(result_path, 'wb') as handle: 
                pickle.dump(config["evaluations"], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving evaluation predictoins into : {result_path}')

        

        





    


     


    