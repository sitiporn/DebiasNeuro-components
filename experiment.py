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
from utils import Intervention, get_overlap_thresholds, group_by_treatment, test_mask
from utils import collect_output_components, report_gpu
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import numpy as np
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)

# from fairseq.data.data_utils import collate_tokens
# from fairseq.models.roberta import RobertaModel


class ExperimentDataset(Dataset):
    def __init__(self, data_path, json_file, upper_bound, lower_bound, encode, num_samples, DEBUG=False) -> None: 
        # combine these two set
        self.encode = encode

        self.premises = {}
        self.hypothesises = {}
        self.labels = {}
        self.intervention = {}
        pair_and_label = []

        sets = {"High-overlap": {}, "Low-overlap": {} }
        nums = {"High-overlap": {}, "Low-overlap": {} }
        
        torch.manual_seed(42)
        
        data_path = os.path.join(data_path, json_file)

        self.df = pd.read_json(data_path, lines=True)
        

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
        print(thresholds)
        
        self.df_exp_set = {"High-overlap": self.get_high_shortcut(),
                           "Low-overlap":  self.get_low_shortcut()}

        
        for do in ["High-overlap", "Low-overlap"]:
            for type in ["entail", "non"]:

                type_selector = self.df_exp_set[do].gold_label == "entailment" if type == "entail" else self.df_exp_set[do].gold_label != "entailment"
                
                sets[do][type] = self.df_exp_set[do][type_selector].reset_index(drop=True)
                nums[do][type] = sets[do][type].shape[0]

        # get minimum size of samples
        self.type_balance = min({min(d.values()) for d in nums.values()})
        self.balance_sets = {}

        # Randomized Controlled Trials (RCTs)
        for do in ["High-overlap", "Low-overlap"]:
            
            self.balance_sets[do] = None
            frames = []

            # create an Empty DataFrame object
            for type in ["entail", "non"]:
                
                # samples data
                ids = list(torch.randint(0, sets[do][type].shape[0], size=(self.type_balance,)))
                sets[do][type] = sets[do][type].loc[ids].reset_index(drop=True)

                #combine type 
                frames.append(sets[do][type])
            
            self.balance_sets[do] =  pd.concat(frames).reset_index(drop=True)
            
            assert self.balance_sets[do].shape[0] == (self.type_balance * 2)

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
        return self.type_balance * 2
    
    def __getitem__(self, idx):

        pair_sentences = {}
        labels = {}

        for do in ["High-overlap", "Low-overlap"]:
            pair_sentences[do] = self.intervention[do].pair_sentences[idx]
            labels[do] = self.labels[do][idx]
        
        return pair_sentences , labels

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

def get_average_activations(path, layers, heads):

    # load all output components 
    with open(path, 'rb') as handle:
        
        # get [CLS] activation 
        q_cls = pickle.load(handle)
        k_cls = pickle.load(handle)
        v_cls = pickle.load(handle)
        ao_cls = pickle.load(handle)
        intermediate_cls = pickle.load(handle)
        out_cls = pickle.load(handle)
        
        attention_data = pickle.load(handle)
        counter = pickle.load(handle)

    # get average of [CLS] activations
    q_cls_avg = {}
    k_cls_avg = {}
    v_cls_avg = {}
    ao_cls_avg = {}
    intermediate_cls_avg = {}
    out_cls_avg = {}
    
    for do in ["High-overlap", "Low-overlap"]:

        q_cls_avg[do] = {}
        k_cls_avg[do] = {}
        v_cls_avg[do] = {}
        ao_cls_avg[do] = {}
        intermediate_cls_avg[do] = {}
        out_cls_avg[do] = {}
         
         # concate all batches
        for layer in layers:

            # compute average over samples
            q_cls_avg[do][layer] = q_cls[do][layer] / counter
            k_cls_avg[do][layer] = k_cls[do][layer] / counter
            v_cls_avg[do][layer] = v_cls[do][layer] / counter

            ao_cls_avg[do][layer] = ao_cls[do][layer] / counter
            intermediate_cls_avg[do][layer] = intermediate_cls[do][layer] / counter
            out_cls_avg[do][layer] = out_cls[do][layer] / counter

    return q_cls_avg, k_cls_avg, v_cls_avg, ao_cls_avg, intermediate_cls_avg,  out_cls_avg

def neuron_intervention(neuron_ids, 
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

        neuron_values = neuron_values.repeat(output.shape[0], output.shape[1], 1)

        output.masked_scatter_(scatter_mask, neuron_values)

    return intervention_hook

def cma_analysis(save_representation_path, save_nie_set_path, model, layers, treatments, heads, tokenizer, experiment_set, label_maps, num_sampling, DEVICE):

    cls_averages = {}
    pairs = {}
    NIE = {}
    mediators = {}
    counter = 0
    combine_types = []
    dataset = None
    dataloader = None


    cls_averages["Q"], cls_averages["K"], cls_averages["V"], cls_averages["AO"], cls_averages["I"], cls_averages["O"] = get_average_activations(save_representation_path, 
                                    layers, 
                                    heads)

    
    # get the whole set of validation 
    pairs["entail"] = list(experiment_set.df[experiment_set.df.gold_label == "entailment"].pair_label)
    pairs["non"] = list(experiment_set.df[experiment_set.df.gold_label != "entailment"].pair_label)
    
    torch.manual_seed(42)
    
    if not os.path.isfile(save_nie_set_path):

        for type in ["entail","non"]:
            
            # samples data
            ids = list(torch.randint(0, len(pairs[type]), size=(num_sampling//2,)))
            pairs[type] = np.array(pairs[type])[ids,:].tolist()
            combine_types.extend(pairs[type])

        # Bug: checking dataset

        # dataset = [([premise, hypo], label) for idx, (premise, hypo, label) in enumerate(pairs['entailment'])]
        dataset = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(combine_types)]
        
        dataloader = DataLoader(dataset, batch_size=32)
        
        with open(save_nie_set_path, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Done saving NIE dataset into pickle !")
    
    else:
        
        with open(save_nie_set_path, 'rb') as handle:
            
            dataset = pickle.load(handle)
            dataloader = pickle.load(handle)
            
            print(f"loading nie sets from pickle {save_nie_set_path} !")

    
    # mediator used to intervene
    mediators["Q"] = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    mediators["K"] = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    mediators["V"] = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    mediators["AO"]  = lambda layer : model.bert.encoder.layer[layer].attention.output
    mediators["I"]  = lambda layer : model.bert.encoder.layer[layer].intermediate
    mediators["O"]  = lambda layer : model.bert.encoder.layer[layer].output

    for batch_idx, (sentences, labels) in enumerate(tqdm(dataloader, desc="DataLoader")):

        premise, hypo = sentences

        pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]

        # distributions = {}
        probs = {}
        
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

        with torch.no_grad(): 
            # Todo: generalize to distribution if the storage is enough
            probs['null'] = F.softmax(model(**inputs).logits , dim=-1)[:, label_maps["entailment"]]

        counter += probs['null'].shape[0] 
        
        # report_gpu()
        
        # To store all positions
        probs['intervene'] = {}

        # run one full neuron intervention experiment
        for do in treatments: # ['do-treatment','no-treatment']
            # Todo : add qeury, key and value 

            NIE[do] = {}
            probs['intervene'][do] = {}

            for component in ["Q","K","V","AO","I","O"]: 
                if  component not in NIE[do].keys():
                    NIE[do][component] = {}
                
            # get each prediction for each single nueron intervention
            for layer in tqdm(layers, desc="layers"):
                
                # probs['intervene'][do][layer] = {}
                 
                # neuron_ids = [*range(0, ao_cls_avg[do][layer].shape[0], 1)]
                for component in tqdm(["Q","K","V","AO","I","O"], desc="Components"): 

                    if  layer not in NIE[do][component].keys(): 
                        NIE[do][component][layer] = {}
                
                    for neuron_id in range(cls_averages[component][do][layer].shape[0]):

                        # NIE[do][layer][neuron_id] = {}
                        
                        # select layer to register and input which neurons to intervene
                        hooks = [] 

                        hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(neuron_ids = [neuron_id], value = cls_averages[component][do][layer])))
                        
                        with torch.no_grad(): 
                            # probs['intervene'][layer][neuron_id] 
                            intervene_probs = F.softmax(model(**inputs).logits , dim=-1)[:, label_maps["entailment"]]

                        # report_gpu()
                        
                        # compute NIE
                        # Eu [ynull,zset-gender (u) (u)/ynull (u) − 1].
                        if neuron_id not in NIE[do][component][layer].keys():
                            NIE[do][component][layer][neuron_id] = 0

                        NIE[do][component][layer][neuron_id] += torch.sum( (intervene_probs / probs['null'])-1, dim=0)
                        
                        for hook in hooks: hook.remove() 
                        
                        # print(f"batch{batch_idx}, layer : {layer}, {component}, Z :{neuron_id}")
    
    with open(f'../pickles/NIE_{treatments[0]}_{layers[0]}.pickle', 'wb') as handle:
        pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(cls_averages, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_top_k(layers, treatments, top_k=5):
        
 
    # compute average NIE
    for do in treatments:
        
        for layer in layers:
            
            ranking_nie = {}
        
            read_path = f'../pickles/NIE_{do}_{layer}.pickle'
            
            with open(read_path, 'rb') as handle:
                NIE = pickle.load(handle)
                counter = pickle.load(handle)
                cls_averages = pickle.load(handle)
                print(f"current : {read_path}")
            
            for component in ["Q","K","V","AO","I","O"]: 
            
                for neuron_id in range(cls_averages[component][do][layer].shape[0]):
            
                    NIE[do][component][layer][neuron_id] = NIE[do][component][layer][neuron_id] / counter

                    ranking_nie[component + "-" + str(neuron_id)] = NIE[do][component][layer][neuron_id].to('cpu')
            
                    # Todo: get component and neuron_id and value 
            
            top_neurons = dict(sorted(ranking_nie.items(), key=operator.itemgetter(1), reverse=True)[:5])
            
            with open(f'../pickles/top_neuron_{do}_{layer}.pickle', 'wb') as handle:
                pickle.dump(top_neurons, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Done saving top neurons into pickle !")


def main():

    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--layer",
                        type=int,
                        default=-1,
                        required=False,
                        help="layers to intervene")
    
    parser.add_argument("--treatment",
                        type=bool,
                        default=False,
                        required=False,
                        help="high or low overlap")

    parser.add_argument("--analysis",
                        type=bool,
                        default=False,
                        required=False,
                        help="compute cma analysis")
    
    parser.add_argument("--top_k",
                        type=bool,
                        default=False,
                        required=False,
                        help="get top K analysis")
    
    
    args = parser.parse_args()

    select_layer = [args.layer]
    do = args.treatment
    is_analysis = args.analysis
    is_topk = args.top_k


    DEBUG = True
    
    valid_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_dev_matched.jsonl'
    
    save_representation_path = '../pickles/activated_components.pickle'
    save_nie_set_path = '../pickles/nie_samples.pickle'
    collect_representation = True
    
    # used to compute nie scores
    num_samples = 2000
    
    # percent threshold of overlap score
    upper_bound = 80
    lower_bound = 20
    
    
    label_maps = {"contradiction": 0 , "entailment" : 1, "neutral": 2}

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased-mnli/")
    model = AutoModelForSequenceClassification.from_pretrained("../bert-base-uncased-mnli/")
    model = model.to(DEVICE)
        
    # Todo: generalize for every model 
    layers = [*range(0, 12, 1)]
    heads =  [*range(0, 12, 1)]
    
    # using same seed everytime we create HOL and LOL sets 
    experiment_set = ExperimentDataset(valid_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                             encode = tokenizer,
                             num_samples = num_samples
                            )

    
    dataloader = DataLoader(experiment_set, 
                        batch_size = 64,
                        shuffle = False, 
                        num_workers=0)

     
    if not os.path.isfile(save_representation_path):
    
        # Todo:  fixing hardcode of vocab.bpe and encoder.json for roberta fairseq
        collect_output_components(model = model,
                                dataloader = dataloader,
                                tokenizer = tokenizer,
                                DEVICE = DEVICE,
                                layers = layers,
                                heads = heads)
        
        print(f"done with saving representation into {save_representation_path}")
        
        return
    
    else:
        print(f"HOL and LOL representation in {save_representation_path}")
 
    if do: 
        mode = ["High-overlap"] 
    else:
        mode = ["Low-overlap"]

    if is_analysis: 
        print(f"perform Causal Mediation analysis...")
        
        cma_analysis(save_representation_path = save_representation_path,
                    save_nie_set_path = save_nie_set_path,
                    model = model,
                    layers = select_layer,
                    treatments = mode,
                    heads  =  heads,
                    tokenizer = tokenizer,
                    experiment_set = experiment_set,
                    label_maps = label_maps,
                    num_sampling = num_samples,
                    DEVICE = DEVICE)

    if is_topk:
        print(f"perform ranking top neurons...")
        get_top_k(select_layer, treatments=mode)
    
    # prunning(model = model,
    #          layers= [0, 1, 2, 3, 4])
    
    # Todo: collect output of every compontent in model 
    # 1. do-treatment : high overlap scores
    # 2. no-treament : low overlap scores

    
    # neuron_intervention(model = model,
    #                     tokenizer = tokenizer,
    #                     layers = [0, 1, 2, 3, 4],
    #                     neurons = [1,2,3],
    #                     dataloader = dataloader) 
                    
    

if __name__ == "__main__":
    main()

