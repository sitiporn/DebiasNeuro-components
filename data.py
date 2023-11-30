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
#from nn_pruning.patch_coordinator import (
#    SparseTrainingArguments,
#    ModelPatchingCoordinator,
#)
from utils import  EncoderParams
from cma_utils import get_overlap_thresholds, group_by_treatment, get_hidden_representations
from intervention import Intervention, neuron_intervention, get_mediators
from utils import get_ans, compute_acc
from utils import get_num_neurons, get_params, relabel, give_weight
from torch.optim import Adam
from transformers import AutoTokenizer, BertForSequenceClassification
from functools import partial
from utils import load_model

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
        
        # Todo: generalize dev apth and json file to  mateched
        self.inputs = {}
        # dev_path = "debias_fork_clean/debias_nlu_clean/data/nli/"
        # dev_path = os.path.join(os.path.join(dev_path, file))
        self.dev_name = list(json_file.keys())[0] if isinstance(json_file, dict) else json_file.split('_')[0] + '_' + json_file.split('_')[-1].split('.')[0]
        data_path = os.path.join(data_path, json_file[self.dev_name] if isinstance(json_file, dict) else json_file)
        self.df = pd.read_json(data_path, lines=True)
            
        if self.dev_name == 'reweight': self.df['weight_score'] = self.df[['gold_label', 'bias_probs']].apply(lambda x: give_weight(*x), axis=1)
        if '-' in self.df.gold_label.unique(): 
            self.df = self.df[self.df.gold_label != '-'].reset_index(drop=True)
        if self.dev_name == 'hans' or self.dev_name == 'heuristics_set': 
            self.df['sentence1'] = self.df.premise 
            self.df['sentence2'] = self.df.hypothesis
        
        for  df_col in list(self.df.keys()): self.inputs[df_col] = self.df[df_col].tolist()
        # self.premises = self.df.sentence1.tolist() if self.dev_name == "mismatched" else self.df.premise.tolist()
        # self.hypos = self.df.sentence2.tolist() if self.dev_name == "mismatched" else self.df.hypothesis.tolist()
        # self.labels = self.df.gold_label.tolist()

    def __len__(self): return self.df.shape[0]

    def __getitem__(self, idx):
        
        return tuple([self.inputs[df_col][idx] for  df_col in list(self.df.keys())])

        # return pair_sentence , label

def preprocss(df):
    if '-' in df.gold_label.unique(): 
        df = df[df.gold_label != '-'].reset_index(drop=True)

    return df

class CustomDataset(Dataset):
    def __init__(self, config, label_maps, data_name = 'train_data', DEBUG=False) -> None: 
        df = pd.read_json(os.path.join(config['data_path'], config[data_name]), lines=True)
        df = preprocss(df)
        if "bias_probs" in df.columns:
          df_new = df[['sentence1', 'sentence2', 'gold_label','bias_probs']]  
        else:
            df_new = df[['sentence1', 'sentence2', 'gold_label']]
        df_new.rename(columns = {'gold_label':'label'}, inplace = True)
        self.label_maps = label_maps
        df_new['label'] = df_new['label'].apply(lambda label_text: self.to_label_id(label_text))
        from datasets import Dataset as HugginfaceDataset
        self.dataset = HugginfaceDataset.from_pandas(df_new)
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
        self.tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
    def to_label_id(self, text_label): 
        return self.label_maps[text_label]
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence1"], examples["sentence2"], truncation=True) 
   
    def __len__(self): 
        return self.tokenized_datasets.shape[0]
    
    def __getitem__(self, idx):
        return self.tokenized_datasets[idx]


def get_conditional_inferences(config, do,  model_path, model, method_name, NIE_paths, counterfactual_paths, tokenizer, DEVICE, seed=None , debug = False):
    """ getting inferences while modifiying activations on dev-matched/dev-mm/HANS"""
    import copy
    from cma import scaling_nie_scores
    layer = config['layer']
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    seed = config['seed'] if seed is None else seed
    seed = str(seed)
    layers = config['layers']  if config['computed_all_layers'] else [config['layer']]
    # Todo: fix loading model using deep copy method
    # load model
    if model_path is not None: 
        _model = load_model(path= model_path, model=copy.deepcopy(model))
    else:
        _model = copy.deepcopy(model)
        print(f'using original model : {config["model_name"]}')
    
    print(f'{config["dev-name"]} : compute on {config["dev_json"]}')
    mediators  = get_mediators(_model)
    params  = get_params(config) 
    digits = [ len(str(epsilon).split('.')[-1]) for epsilon in params['epsilons'] ]
    total_neurons = get_num_neurons(config)
    epsilons = params['epsilons']
    if not isinstance(epsilons, list): epsilons = epsilons.tolist()
    
    dev_set = Dev(config['dev_path'], config['dev_json'])
    dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)
    
    select_neuron_mode = 'percent' if config['k'] is not None  else config['weaken_rate'] if config['weaken_rate'] is not None else 'neurons'
    top_k_mode =  'percent' if config['range_percents'] else ('k' if config['k'] else 'neurons')
    path = f'../pickles/top_neurons/{method_name}/top_neuron_{seed}_{select_neuron_mode}_{do}_all_layers.pickle' if config['computed_all_layers'] else f'../pickles/top_neurons/top_neuron_{seed}_{do}_{layer}_.pickle'
    
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else ( [config['masking_rate']] if config['masking_rate'] is not None else list(top_neuron.keys()))
    if config['masking_rate_search']: num_neuron_groups = params['percent']
    top_k_mode =  'percent' if config['range_percents'] else ( 'k' if config['k'] else 'neurons')

    nie_table_df = scaling_nie_scores(config, method_name, NIE_paths, debug=False)
    m = {row['Neuron_ids']: row['M_MinMax'] for index, row in nie_table_df.iterrows()} 

    # cls = get_hidden_representations(counterfactual_paths, layers, config['is_group_by_class'], config['is_averaged_embeddings'])
    cls = get_hidden_representations(counterfactual_paths, method_name, seed, layers, config['is_group_by_class'], config['is_averaged_embeddings'])
    cls = cls[seed]
    # iterate throught all weaken rates
    for eps_id, epsilon in enumerate(t := tqdm(epsilons)): 
        prediction_path = f'../pickles/prediction/{method_name}/seed_{seed}/' 
        if not os.path.isdir(prediction_path): os.mkdir(prediction_path) 
        # where to save modifying activation results
        prediction_path =  os.path.join(prediction_path, f'esp-{round(epsilon, digits[eps_id])}')
        if not os.path.isdir(prediction_path): os.mkdir(prediction_path) 
        t.set_description(f"epsilon : {epsilon} , prediction path : {prediction_path}")
        # iterate through all top-k 
        for masking_rate in (n:= tqdm(num_neuron_groups)):
            n.set_description(f"masking_rate : {masking_rate}")
            cur_num_neurons = nie_table_df.shape[0] * masking_rate
            cur_num_neurons = int(cur_num_neurons)
            acc = {}
            
            if config['computed_all_layers']:
                layer_ids  = [neuron['Neuron_ids'].split('-')[1] for row, neuron in nie_table_df[:cur_num_neurons].iterrows()]
                components = [neuron['Neuron_ids'].split('-')[2] for row, neuron in nie_table_df[:cur_num_neurons].iterrows()]
                neuron_ids = [neuron['Neuron_ids'].split('-')[3] for row, neuron in nie_table_df[:cur_num_neurons].iterrows()]
            else:
                # Todo: customize for specific layers
                components = [neuron['Neuron_ids'].split('-')[0] for row, neuron in nie_table_df.iterrows()]
                neuron_ids = [neuron['Neuron_ids'].split('-')[1] for row, neuron in nie_table_df.iterrows()]
                layer_ids  = [layer] * len(components)

            if config['single_neuron']: 
                layer_ids  =  [layer]
                components = [components[0]]
                neuron_ids = [neuron_ids[0]]
                raw_distribution_path = f'raw_distribution_{do}_L{layer}_{component}_{config["intervention_type"]}_{config["dev-name"]}.pickle'  
            else:
                if config['computed_all_layers']:
                    raw_distribution_path = f'raw_distribution_{do}_all_layers_{masking_rate}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'  
                else:
                    raw_distribution_path = f'raw_distribution_{do}_L{layer}_{masking_rate}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'
            
            distributions = {}
            losses = {}
            golden_answers = {}
            
            for mode in ["Null", "Intervene"]: 
                distributions[mode] = []
                golden_answers[mode] = []
                losses[mode] = []
            for batch_idx, (inputs) in enumerate(b:=tqdm(dev_loader)):
                b.set_description(f"batch_idx: {batch_idx}")
                cur_inputs = {} 
                for idx, (cur_inp, cur_col) in enumerate(zip(inputs, list(dev_set.df.keys()))): cur_inputs[cur_col] = cur_inp
                pair_sentences = [[premise, hypo] for premise, hypo in zip(cur_inputs['sentence1'], cur_inputs['sentence2'])]
                pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
                pair_sentences = {k: v.to(DEVICE) for k,v in pair_sentences.items()}

                # ignore label_ids when running experiment on hans
                label_ids = torch.tensor([config['label_maps'][label] for label in cur_inputs['gold_label']]) if config['dev-name'] != 'hans' else None
                scalers = cur_inputs['weight_score'] if config["dev-name"] == 'reweight' else 1

                # ignore label_ids when running experiment on hans
                if label_ids is not None: label_ids = label_ids.to(DEVICE)
                if config['dev-name'] == 'reweight': scalers = scalers.to(DEVICE)
                # mediator used to intervene
                cur_dist = {}
                cur_loss = {}
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
                        outs =  _model(**pair_sentences, labels= label_ids if config['dev-name'] != 'hans' else None)

                        cur_dist[mode] = F.softmax(outs.logits , dim=-1)
                        cur_loss[mode] = outs.loss

                        if config['dev-name'] != 'hans':
                            loss = criterion(outs.logits, label_ids)
                            test_loss = torch.mean(loss)
                            assert (test_loss - cur_loss[mode]) < 1e-6
                            if debug: print(f"test loss : {test_loss},  BERT's loss : {cur_loss[mode]}")
                            if debug: print(f"Before reweight : {test_loss}")
                            
                            loss =  scalers * loss
                            
                            if debug: print(f"After reweight : {torch.mean(loss)}")

                    if mode == "Intervene": 
                        for hook in hooks: hook.remove() 

                    distributions[mode].extend(cur_dist[mode])
                    golden_answers[mode].extend(label_ids if label_ids is not None else cur_inputs['gold_label']) 
                    if config['dev-name'] != 'hans': losses[mode].extend(loss) 

            raw_distribution_path = os.path.join(prediction_path,  raw_distribution_path)

            with open(raw_distribution_path, 'wb') as handle: 
                pickle.dump(distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(golden_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving distributions and labels into : {raw_distribution_path}')
            if dev_set.dev_name != 'hans': 
                acc[masking_rate] = compute_acc(raw_distribution_path, config["label_maps"])

                eval_path  = get_eval_path(config, select_neuron_mode, method_name, seed, epsilon, digits, eps_id, masking_rate, do)
            
                with open(eval_path,'wb') as handle:
                    pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"saving all accuracies into {eval_path} ")

def get_eval_path(config, select_neuron_mode, method_name, seed, epsilon, digits, eps_id, value, do):
    eval_path =  f'../pickles/evaluations/{method_name}/'
    if not os.path.isdir(eval_path): os.mkdir(eval_path)
    eval_path =  os.path.join(eval_path, f'seed_{seed}')
    if not os.path.isdir(eval_path): os.mkdir(eval_path)
    eval_path =  os.path.join(eval_path, f'esp-{round(epsilon, digits[eps_id])}')
    if not os.path.isdir(eval_path): os.mkdir(eval_path)
    
    if config["masking_rate"]:
        pickle_path = f'{select_neuron_mode}_{value}_{do}_{config["intervention_type"]}_{config["dev-name"]}.pickle' 
    else: 
        pickle_path = f'{select_neuron_mode}_{do}_{config["intervention_type"]}_{config["dev-name"]}.pickle'
    
    eval_path = os.path.join(eval_path, pickle_path)
    
    return eval_path

def get_masking_value(config):
    import glob
    # seed = config['seed']
    value = 0.05
    
    for seed in [42, 3099, 3785, 3990,  409]:
        eval_paths = glob.glob(f"../pickles/evaluations/seed_{seed}/**/*.pickle")
        print(len(eval_paths))
        scores = {}
        print(f'********** seed : {seed} ************')
        for path in eval_paths:
            with open(path,'rb') as handle:
                acc = pickle.load(handle)
            scores[path.split('/')[4]] = acc[value]['Intervene']['all']
            # print(path)
        sorted_scores = dict(sorted(scores.items()))
        # for k,v in  sorted_scores.items():
        #     print(f'{k}:{v*100:.2f}')
        # print(f"Null : {acc[value]['Null']['all']*100:.2f}, Intervene :{acc[value]['Intervene']['all']*100:.2f}")
        cur_best_key = list(sorted_scores)[-1]
        best_val = float(cur_best_key.split('-')[-1])
        # cur_digits = len(str(best_val).split('.')[-1])
        print(f"Null : {acc[value]['Null']['all']*100:.2f}, Intervene at {best_val}:{scores[cur_best_key]*100:.2f}")
    # get new high and low value unitil up to max_num_digits 
    # get digits
    # get of new 
    # add digits by one
    # 0.1 / 2 #  -> diff 1 * 10e-cu_digits / 2
    # 0.9 + diff , 
    # 

#  for a masking representation experiment 
def convert_to_text_ans(config, top_neuron, method_name, params, do, seed=None, text_answer_path = None, raw_distribution_path = None):
    
    """changing distributions into text anaswers on hans set
    
    Keyword arguments:
    raw_distribution_path --  to read distribution 
    text_answer_path  -- write text into text file
    
    """
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else ( [config['masking_rate']] if config['masking_rate'] else list(top_neuron.keys()))
    if config['masking_rate_search']: num_neuron_groups = params['percent']
    
    digits = [ len(str(epsilon).split('.')[-1]) for epsilon in params['epsilons'] ]
    layer = config['layer']
    epsilons = params['epsilons']
    seed = config['seed'] if seed is None else seed
    seed = str(seed)

    # select neuron group type 
    topk_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
    prediction_path = f'../pickles/prediction/{method_name}' 
    
    for idx, epsilon in enumerate(t := tqdm(epsilons)):  
        # read pickle file used to interpret as text answers later
        epsilon_path = f'esp-{round(epsilon, digits[idx])}'

        for neurons in (n:= tqdm(num_neuron_groups)):

            # dont touch this 
            raw_distribution_path = f'raw_distribution_{do}_all_layers_{neurons}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'  
            raw_distribution_path = os.path.join(os.path.join(prediction_path, f'seed_{seed}',epsilon_path),  raw_distribution_path)
            
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
                
                if config["single_neuron"]:
                    component = [neuron.split('-')[2 if layer == -1 else 0] for neuron, v in top_neuron[neurons].items()][0]
                    text_answer_path = f'txt_answer_{topk_mode}_{mode}_L{layer}_{component}_{config["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
                else:
                    if config['computed_all_layers']:
                        text_answer_path = f'txt_answer_{topk_mode}_{mode}_all_layers_{neurons}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
                    else:
                        text_answer_path = f'txt_answer_{topk_mode}_{mode}_L{layer}_{neurons}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

                text_answer_path  = os.path.join(os.path.join(prediction_path,f'seed_{seed}' ,epsilon_path), text_answer_path)

                with open(text_answer_path, "w") as fobj:

                    headers = ['pairID', 'gold_label']

                    fobj.write(f'{headers[0]}' + "," + f'{headers[1]}' +"\n")
                    
                    for sample_id, ans in enumerate(text_answers[mode]):

                        fobj.write(f"ex{sample_id}" + "," + ans +"\n")

                    print(f"saving text answer's bert predictions {mode}: {text_answer_path}")

        

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

def get_condition_inference_scores(config, model, method_name, seed=None): 
    """ getting scores on modifiying activations on dev-matched, dev-mm and challenge set(HANS)"""
    eval_path = f'../pickles/evaluations/{method_name}/'
    prediction_path = f'../pickles/prediction/{method_name}/' 
    # top_mode =  'percent' if config['range_percents'] else ('k' if config['k'] else 'neurons')
    seed = config['seed'] if seed is None else str(seed)
    layer = config["layer"]
    k = config['k']
    num_neurons = None
    from cma import get_topk
    topk = get_topk(config, k=k, num_top_neurons=num_neurons)
    key = list(topk.keys())[0] # masking model eg. percent, k, num_neurons
    do = config["eval"]["do"]
    if config['computed_all_layers']:  
        neuron_path = f'../pickles/top_neurons/{method_name}/top_neuron_{seed}_{key}_{do}_all_layers.pickle' 
    else:                                                       
        neuron_path = f'../pickles/top_neurons/{method_name}/top_neuron_{seed}_{key}_{do}_{layer}.pickle'
    with open(neuron_path, 'rb') as handle: 
        top_neuron = pickle.load(handle)
        print(f'neuron path: {neuron_path}')
    # with open(f'../pickles/top_neurons/top_neuron_{seed}_{key}_{do}_all_layers.pickle', 'wb') as handle: top_neuron = pickle.load(handle)
    topk_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
    params = get_params(config)
    digits = [ len(str(epsilon).split('.')[-1]) for epsilon in params['epsilons'] ]

    # ********** follow **********
    # ********** original function **********
    # required distribution of hans
    if config['to_text']: convert_to_text_ans(config, top_neuron, method_name, params, do, seed)
 
    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else ([config['masking_rate']] if config['masking_rate'] else list(top_neuron.keys()))
    if config['masking_rate_search']: num_neuron_groups = params['percent']

    for idx, epsilon in enumerate(t := tqdm(params['epsilons'])):  

        epsilon_path = f'esp-{round(epsilon, digits[idx])}'
        t.set_description(f"epsilon : {epsilon} ")
        for group in num_neuron_groups:
            hans_scores = {}
            for mode in ['Null','Intervene']:
                # after convert to txt answer 
                text_answer_path = f'txt_answer_{topk_mode}_{mode}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
                result_path = f'result_{topk_mode}_{mode}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

                if config['eval']['all_layers']: text_answer_path = f'txt_answer_{topk_mode}_{mode}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
                if config['eval']['all_layers']: result_path = f'result_{topk_mode}_{mode}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

                config['evaluations'][group] = {}
                # text_answer_path = os.path.join(os.path.join(prediction_path, epsilon_path),  text_answer_path)
                text_answer_path  = os.path.join(os.path.join(prediction_path, f'seed_{seed}' ,epsilon_path), text_answer_path)
                result_path = os.path.join(os.path.join(eval_path, f'seed_{seed}',epsilon_path),  result_path)
                # vv = '../pickles/prediction/seed_None/v-0.9/txt_answer_percent_Intervene_all_layers_0.05-k_High-overlap_weaken_hans.txt'
                # get_hans_result(cur_raw_distribution_path, config)
                convert_text_to_hans_scores(text_answer_path, config, result_path, group)
                hans_scores[mode] = get_avg_score(result_path)

            valid_result =  f'{topk_mode}_{group}_{do}_{config["intervention_type"]}_matched.pickle'
            dev_mm_result = f'{topk_mode}_{group}_{do}_{config["intervention_type"]}_mismatched.pickle'
            valid_result =   os.path.join(eval_path, f'seed_{seed}', epsilon_path, valid_result)
            dev_mm_result =  os.path.join(eval_path, f'seed_{seed}', epsilon_path, dev_mm_result)
            # f'../pickles/evaluations/seed_None/v-0.9/percent_0.05_High-overlap_weaken_matched.pickle'
            # f'../pickles/evaluations/seed_None/v-0.9/percent_0.05_High-overlap_weaken_mismatched.pickle'
            with open(valid_result,'rb') as handle: valid_acc = pickle.load(handle)
            with open(dev_mm_result,'rb') as handle: dev_mm_acc = pickle.load(handle)
            print(f"*********** esp: {epsilon}, masking rate : {group} **************")
            print(f"Matched :")
            print(f"-- Intervene  : {valid_acc[group]['Intervene']['all']*100:.2f}")
            print(f"-- Null : {valid_acc[group]['Null']['all']*100:.2f}")
            print(f"Dev-mm :")
            print(f"-- Intervene  : {dev_mm_acc[group]['Intervene']['all']*100:.2f}")
            print(f"-- Null : {dev_mm_acc[group]['Null']['all']*100:.2f}")
            print(f'HAN scores : ')
            print(f"-- Intervene  : {hans_scores['Intervene']*100:.2f}")
            print(f"-- Null  : {hans_scores['Null']*100:.2f}")
            # print(f"-- Null : 56.72")
             

def rank_losses(config, do):  

    # get weaken rates parameters
    params, digits = get_params(config)
    total_neurons = get_num_neurons(config)
    epsilons = params['epsilons'] if config["weaken"] is None else [ config["weaken"]]
    epsilons = sorted(epsilons)
    key = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
    layer = config['layer']
    average_losses = {}

    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else ( [config['masking_rate']] if config['masking_rate'] is not None else list(top_neuron.keys()))

    if not isinstance(epsilons, list): epsilons = epsilons.tolist()

    for epsilon in epsilons:

        prediction_path = '../pickles/prediction/' 
        
        prediction_path =  os.path.join(prediction_path, f'v{round(epsilon, digits["epsilons"])}')

        for value in num_neuron_groups:
    
            # dum_path = '../pickles/prediction/v0.5/raw_distribution_neurons_High-overlap_all_layers_0.05-k_weaken_reweight.pickle'
            

            if layer == -1:
                raw_distribution_path = f'raw_distribution_{key}_{do}_all_layers_{value}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'  
            else:
                raw_distribution_path = f'raw_distribution_{key}_{do}_L{layer}_{value}-k_{config["intervention_type"]}_{config["dev-name"]}.pickle'

            
            raw_distribution_path = os.path.join(prediction_path,  raw_distribution_path)

            # Bug fix: when setting specific weaken rate

            with open(raw_distribution_path, 'rb') as handle:
                # get [CLS] activation 
                distributions = pickle.load(handle)
                golden_answers = pickle.load(handle)
                losses = pickle.load(handle)

            """
            consider Intervene-ony mode 
            select the lowest loss  on hans set
            """
            average_losses = torch.mean(torch.tensor(losses['Intervene']))

            print(f"curret loss : {average_losses} on weaken rate : {epsilon}")

def get_analysis(config):
    RESULT_PATH = f'../pickles/performances/'
    name_set = list(config['dev_json'])[0] 
    raw_distribution_path = os.path.join(RESULT_PATH, f'inference_{name_set}.pickle')
    key = 'reweight'

    with open(raw_distribution_path, 'rb') as handle:
        distributions  = pickle.load(handle)
        golden_answers = pickle.load(handle)
        print(f"loading file pickle {raw_distribution_path} !")        

    distributions = torch.stack(distributions[key], dim=0)
    avg_dist = torch.mean(distributions, dim=0)
    print(f'average distribution of each class : {avg_dist}')

def get_all_model_paths(LOAD_MODEL_PATH):
    import pathlib
    seed_path_ind = 3
    num_seeds = 5
    model_files = pathlib.Path(LOAD_MODEL_PATH)
    model_files.rglob('*.bin')
    all_model_files = {} 
    clean_model_files = []
    
    for f in model_files.rglob("*.bin"):
        key = str(f).split("/")[seed_path_ind]
        if  key not in all_model_files.keys(): all_model_files[key] = []
        if 'pytorch_model' not in str(f): continue
        all_model_files[key].append(str(f))
    

    def take_second(elem):
        return elem[1]

    for seed in all_model_files.keys():
        checkpoint_paths = [ (checkpoint.split("/")[4].split('_')[-1], checkpoint) for checkpoint in all_model_files[seed]]
        # load best model is trained up to the end
        checkpoint = sorted(checkpoint_paths, key=take_second, reverse=True)[0]
        clean_model_files.append(checkpoint[-1])
    assert len(clean_model_files) == num_seeds, f"is not {num_seeds} runs"
    return {path.split('/')[3].split('_')[-1]: path for path in clean_model_files}
    
def eval_model(model, config, tokenizer, DEVICE, LOAD_MODEL_PATH, is_load_model=True, is_optimized_set = False):
    """ to get predictions and score on test and challenge sets"""
    distributions = {}
    losses = {}
    golden_answers = {}
    all_paths = get_all_model_paths(LOAD_MODEL_PATH)
    OPTIMIZED_SET_JSONL = config['dev_json']
    # datasets
    IN_DISTRIBUTION_SET_JSONL = 'multinli_1.0_dev_mismatched.jsonl'
    CHALLENGE_SET_JSONL = 'heuristics_evaluation_set.jsonl' 
    RESULT_PATH = f'../pickles/performances/'
    json_sets = [OPTIMIZED_SET_JSONL] if is_optimized_set else [IN_DISTRIBUTION_SET_JSONL, CHALLENGE_SET_JSONL]
    acc_avg = 0
    entail_avg = 0
    contradiction_avg = 0
    neutral_avg = 0
    hans_avg = 0
    computed_acc_count = 0
    computed_hans_count = 0
    seed = str(config['seed'])
    if not config['compute_all_seeds']:  all_paths = {seed: all_paths[seed]}

    for seed, path in all_paths.items():
        if is_load_model:
            from utils import load_model
            model = load_model(path=path, model=model)
        else:
            print(f'Using original model')
        for cur_json in json_sets:
            name_set = list(cur_json.keys())[0] if is_optimized_set else cur_json.split("_")[0] 
            distributions = []
            losses = []
            golden_answers = []
            dev_set = Dev(config['dev_path'] , cur_json)
            dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)
            
            for batch_idx, (inputs) in enumerate( t := tqdm(dev_loader)):
                model.eval()
                cur_inputs = {} 
                t.set_description(f'{name_set} batch_idx {batch_idx}/{len(dev_loader)}')
                for idx, (cur_inp, cur_col) in enumerate(zip(inputs, list(dev_set.df.keys()))): cur_inputs[cur_col] = cur_inp
                # get the inputs 
                pair_sentences = [[premise, hypo] for premise, hypo in zip(cur_inputs['sentence1'], cur_inputs['sentence2'])]
                pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
                pair_sentences = {k: v.to(DEVICE) for k,v in pair_sentences.items()}
                # ignore label_ids when running experiment on hans
                label_ids = torch.tensor([config['label_maps'][label] for label in cur_inputs['gold_label']]) if  'heuristics' not in cur_json  else None
                # ignore label_ids when running experiment on hans
                if label_ids is not None: label_ids = label_ids.to(DEVICE)

                with torch.no_grad(): 
                    # Todo: generalize to distribution if the storage is enough
                    outs =  model(**pair_sentences, labels= label_ids if  'heuristics' not in cur_json else None)
                    distributions.extend(F.softmax(outs.logits.cpu() , dim=-1))
                    golden_answers.extend(label_ids.cpu() if label_ids is not None else cur_inputs['gold_label'])

            cur_raw_distribution_path = os.path.join(RESULT_PATH, f'inference_{name_set}.pickle')
            
            with open(cur_raw_distribution_path, 'wb') as handle: 
                pickle.dump(distributions , handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(golden_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(losses[cur_json.split("_")[0]], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving without condition distribution into : {cur_raw_distribution_path}')
                
            if 'heuristics' not in cur_json: 
                acc = compute_acc(cur_raw_distribution_path, config["label_maps"])
                if 'Null' in acc.keys():
                    acc = acc['Null']
                print(f"overall acc : {acc['all']}")
                print(f"contradiction acc : {acc['contradiction']}")
                print(f"entailment acc : {acc['entailment']}")
                print(f"neutral acc : {acc['neutral']}")

                acc_avg += acc['all']
                entail_avg += acc['entailment']
                neutral_avg += acc['neutral']
                contradiction_avg += acc['contradiction']
                computed_acc_count += 1

            elif config['get_hans_result'] and 'heuristics'in cur_json: 
                cur_hans_score = get_hans_result(cur_raw_distribution_path, config)
                hans_avg += cur_hans_score
                computed_hans_count +=1
                print(f'has score :{cur_hans_score}')
    
    print(f'==================== Avearge scores ===================')
    print(f"average overall acc : {acc_avg / len(all_paths)}")
    print(f"averge contradiction acc : {contradiction_avg / len(all_paths)}")
    print(f"average entailment acc : {entail_avg   / len(all_paths)}")
    print(f"average neutral acc : {neutral_avg /  len(all_paths)}")
    print(f'avarge hans score : { hans_avg  / len(all_paths)}')

def convert_text_to_answer_base(config, raw_distribution_path, text_answer_path):

    text_answers = []
    
    with open(raw_distribution_path, 'rb') as handle: 
        distributions = pickle.load(handle)
        golden_answers = pickle.load(handle)
        print(f'hans loading from : {raw_distribution_path}')
    
    # # convert answers_ids to text answers
    for sample_id in range(len(distributions)):
        text_prediction = get_ans(torch.argmax(distributions[sample_id], dim=-1))
        text_answers.append(text_prediction)

    # # write into text files
    with open(text_answer_path, "w") as fobj:
        headers = ['pairID', 'gold_label']
        fobj.write(f'{headers[0]}' + "," + f'{headers[1]}' +"\n")
        
        for sample_id, ans in enumerate(text_answers):
            fobj.write(f"ex{sample_id}" + "," + ans +"\n")

        print(f"saving text answer's bert predictions: {text_answer_path}")

def get_hans_result(raw_distribution_path, config):
    
    performance_path =  '/'.join(raw_distribution_path.split('/')[:-1])
    text_answer_path =  os.path.join(performance_path, f'hans_text_answers.txt')
    score_path =  os.path.join(performance_path, f'hans_scores.txt')
    if config['to_text']: convert_text_to_answer_base(config, raw_distribution_path, text_answer_path)

    # path to read inferences of model
    # os.path.join(os.path.join(prediction_path, epsilon_path),  text_answer_path)
    # path to write scores
    # result_path = None #os.path.join(os.path.join(eval_path, epsilon_path),  result_path)

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
        if cur_class not in config["evaluations"].keys():  config["evaluations"][cur_class] = {}

        for heuristic in heuristic_list:

            correct = tables['correct'][cur_class][heuristic]
            incorrect = tables['incorrect'][cur_class][heuristic]

            total = correct + incorrect
            percent = correct * 1.0 / total
            print(heuristic + ": " + str(percent))

            config["evaluations"][cur_class][heuristic] = percent

    
    with open(score_path, 'wb') as handle: 
        pickle.dump(config["evaluations"], handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving evaluation predictoins into : {score_path}')

    avg_score = get_avg_score(score_path)
    print(f'average score : {avg_score}')

    return avg_score


def get_avg_score(score_path):

    with open(score_path, 'rb') as handle: 
        current_score = pickle.load(handle)
    
    cur_score = []

    for type in ['entailed','non-entailed']:

        class_score = []

        for score in ['lexical_overlap', 'subsequence','constituent']: class_score.append(current_score[type][score])

        cur_score.append(class_score)

    return torch.mean(torch.mean(torch.Tensor(cur_score), dim=-1),dim=0)

def get_specific_component(splited_name, component_mappings):

    if 'layer' not in splited_name: breakpoint()
    
    layer_id = splited_name[splited_name.index('layer') + 1]
    
    if 'self' in splited_name:  
        component = component_mappings[splited_name[-2]]  # to get Q, K, V
    elif 'attention' in splited_name and 'output' in splited_name: 
        component = component_mappings['attention.output']  
    else:
        component = component_mappings[splited_name[-3]]
    
    return layer_id, component
        
def group_layer_params(layer_params):
    """ group parameter's component of both weight and bias """

    group_param_names = {}
    param_names = list(layer_params.params['weight'].keys())

    for name in param_names:
        component = name.split('-')[0]
        neuron_id = name.split('-')[1]
        if component not in group_param_names.keys(): group_param_names[component] = []
        group_param_names[component].append(int(neuron_id))
    
    return group_param_names

def masking_representation_exp(config, model, method_name, experiment_set, dataloader, NIE_paths, LOAD_MODEL_PATH, counterfactual_paths, tokenizer, DEVICE, is_load_model=True):
    """ """
    # load model 
    # prepare biased neuron positions
    import copy
    from cma_utils import collect_counterfactuals
    model = copy.deepcopy(model)
    all_paths = get_all_model_paths(LOAD_MODEL_PATH)
    mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
    group_counterfactual_paths = {} 
    json_files = ['multinli_1.0_dev_matched.jsonl', 'multinli_1.0_dev_mismatched.jsonl', 'heuristics_evaluation_set.jsonl']
    dataset_names = ['matched', 'mismatched', 'hans']
    if not config['compute_all_seeds']: all_paths = {str(config["seed"]):all_paths[str(config['seed'])]} 

    for path in counterfactual_paths: 
        cur_seed = path.split('/')[3]
        if  cur_seed not in group_counterfactual_paths.keys(): group_counterfactual_paths[cur_seed] = []
        group_counterfactual_paths[cur_seed].append(path)
    
    for seed, path in all_paths.items():
        hooks = []
        # model_path = config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])] 
        model_path = path
        config["dev_json"] = {}
        config['dev-name'] = None

        for dataset_name,  json_file in zip(dataset_names, json_files):
            config['dev-name'] = dataset_name
            config["dev_json"][dataset_name] = json_file
            get_conditional_inferences(config, mode[0], model_path, model, method_name, NIE_paths, group_counterfactual_paths[f'seed_{seed}'], tokenizer, DEVICE, seed, debug = False)
            config["dev_json"].pop(f'{dataset_name}')
        get_condition_inference_scores(config, model, method_name, seed)
        
def convert_text_to_hans_scores(text_answer_path, config, result_path, group): 
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
        pickle.dump(config["evaluations"][group], handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving evaluation prediction {group} into : {result_path}')






