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
from utils import collect_output_components , report_gpu, trace_counterfactual
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
from data import ExperimentDataset, Dev, get_predictions, print_config
from intervention import intervene, high_level_intervention
from analze import cma_analysis, compute_embedding_set, get_distribution, get_top_k
from utils import debias_test

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
    
    parser.add_argument("--distribution",
                        type=bool,
                        default=False,
                        required=False,
                        help="get top distribution")
    
    parser.add_argument("--embedding_summary",
                        type=bool,
                        default=False,
                        required=False,
                        help="get average embeddings")
    
    parser.add_argument("--get_counterfactual",
                        type=bool,
                        default=False,
                        required=False,
                        help="get average embeddings")
    
    parser.add_argument("--trace",
                        type=bool,
                        default=False,
                        required=False,
                        help="tracing counterfactual")
    parser.add_argument("--debias",
                        type=bool,
                        default=False,
                        required=False,
                        help="debias component")
    
    parser.add_argument("--get_prediction",
                        type=bool,
                        default=False,
                        required=False,
                        help="get distributions")
    
    parser.add_argument('--dev_name', 
                        type=str, 
                        help='optional filename', 
                        default="matched")

    
    
    args = parser.parse_args()

    # +++++++++++ read  CLI configs ++++++++++
    select_layer = [args.layer]
    do = args.treatment
    is_analysis = args.analysis
    is_topk = args.top_k
    distribution = args.distribution
    getting_counterfactual = args.get_counterfactual
    embedding_summary = args.embedding_summary
    is_traced = args.trace
    is_prediction = args.get_prediction
    debias = args.debias
    dev_set_name = args.dev_name

    DEBUG = True
    debug = False # for tracing top counterfactual 
    
    # ++++++ select type of counterfactual representatoins +++++++++
    is_group_by_class =   False
    is_averaged_embeddings =   True
    intervention_type = "remove" # ["remove", "neg"]
    upper_bound = 95
    lower_bound = 5
    torch.manual_seed(42)
    collect_representation = True
    mode = ["High-overlap"]  if do else  ["Low-overlap"] 
    
    counterfactual_paths = []
    NIE_paths = []
    is_NIE_exist = []
    is_counterfactual_exist = []
    
    # +++ Compute nie scores ++ 
    num_samples = 300 #3000
    label_maps = {"contradiction": 0 , "entailment" : 1, "neutral": 2}
    layers = [*range(0, 12, 1)]
    heads =  [*range(0, 12, 1)]
         
    # +++++++++++++  experiment set +++++++++++++++
    k = None # percent
    num_top_neurons = 60 #  the number of neurons 
    save_nie_set_path = f'../pickles/class_level_nie_{num_samples}_samples.pickle' if is_group_by_class else f'../pickles/nie_{num_samples}_samples.pickle'
    dev_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    exp_json = 'multinli_1.0_dev_matched.jsonl'
    dev_json = {}
    
    if dev_set_name =='mismatched': dev_json['mismatched'] = 'multinli_1.0_dev_mismatched.jsonl'
    elif dev_set_name == 'hans':    dev_json['hans'] = 'heuristics_evaluation_set.jsonl' 
    elif dev_set_name == 'matched': dev_json['matched'] = 'multinli_1.0_dev_matched.jsonl'

    geting_counterfactual_paths(counterfactual_paths,
                                is_counterfactual_exist,
                                is_averaged_embeddings,
                                is_group_by_class)

    geting_NIE_paths(NIE_paths,
                    select_layer,
                    mode,
                    counterfactual_paths,
                    is_NIE_exist,
                    is_averaged_embeddings,
                    is_group_by_class)
    
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased-mnli/")
    model = AutoModelForSequenceClassification.from_pretrained("../bert-base-uncased-mnli/")
    model = model.to(DEVICE)
        
    # Todo: generalize for every model 
    
    
    # using same seed everytime we create HOL and LOL sets 
    experiment_set = ExperimentDataset(dev_path,
                             exp_json,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                             encode = tokenizer,
                             is_group_by_class = is_group_by_class,
                             num_samples = num_samples
                            )

    dataloader = DataLoader(experiment_set, 
                        batch_size = 32,
                        shuffle = False, 
                        num_workers=0)


    if getting_counterfactual:
    
        collect_output_components(model = model,
                                 counterfactual_paths = counterfactual_paths,
                                 experiment_set = experiment_set,
                                 dataloader = dataloader,
                                 tokenizer = tokenizer,
                                 DEVICE = DEVICE,
                                 layers = layers,
                                 heads = heads,
                                 is_averaged_embeddings = is_averaged_embeddings)

    
    print_config(getting_counterfactual, 
                exp_json,
                dev_json,
                is_group_by_class,
                is_averaged_embeddings,
                upper_bound,
                lower_bound,
                num_samples,
                intervention_type,
                k,
                num_top_neurons,
                is_counterfactual_exist,
                counterfactual_paths)

    if not os.path.isfile(save_nie_set_path):

        combine_types = []
        pairs = {}

        if is_group_by_class:
            
            nie_dataset = {}
            nie_loader = {}

            for type in ["contradiction","entailment","neutral"]:
            
                # get the whole set of validation 
                pairs[type] = list(experiment_set.df[experiment_set.df.gold_label == type].pair_label)
                
                # samples data
                ids = list(torch.randint(0, len(pairs[type]), size=(num_samples //3,)))
                pairs[type] = np.array(pairs[type])[ids,:].tolist()
                
                nie_dataset[type] = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(pairs[type])]
                nie_loader[type] = DataLoader(nie_dataset[type], batch_size=32)

        else:

            # balacing nie set across classes
            for type in ["contradiction","entailment","neutral"]:
            
                # get the whole set of validation 
                pairs[type] = list(experiment_set.df[experiment_set.df.gold_label == type].pair_label)
                
                # samples data
                ids = list(torch.randint(0, len(pairs[type]), size=(num_samples //3,)))
                
                pairs[type] = np.array(pairs[type])[ids,:].tolist()
                combine_types.extend(pairs[type])

            # dataset = [([premise, hypo], label) for idx, (premise, hypo, label) in enumerate(pairs['entailment'])]
            nie_dataset = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(combine_types)]
            nie_loader = DataLoader(nie_dataset, batch_size=32)
        
        with open(save_nie_set_path, 'wb') as handle:
            pickle.dump(nie_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(nie_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Done saving NIE set  into {save_nie_set_path} !")

    if is_analysis:  
        
        print(f"perform Causal Mediation analysis...")
        
        cma_analysis(counterfactual_paths = counterfactual_paths,
                    save_nie_set_path = save_nie_set_path,
                    model = model,
                    layers = select_layer,
                    treatments = mode,
                    heads  =  heads,
                    tokenizer = tokenizer,
                    experiment_set = experiment_set,
                    label_maps = label_maps,
                    is_averaged_embeddings = is_averaged_embeddings,
                    is_group_by_class = is_group_by_class,
                    DEVICE = DEVICE,
                    DEBUG = True)

    if is_topk:

        print(f"perform ranking top neurons...")
        
        if sum(is_NIE_exist) == len(is_NIE_exist):
            get_top_k(NIE_paths, select_layer, treatments=mode, num_top_neurons=num_top_neurons)
        else:
            print("NIE is not enought to get top k")
            return

    if embedding_summary:

        compute_embedding_set(experiment_set, model, tokenizer, label_maps, DEVICE, is_group_by_class = is_group_by_class)
        # get_embeddings(experiment_set, model, tokenizer, label_maps, DEVICE)

    if distribution:
        
        get_distribution(save_nie_set_path, experiment_set, tokenizer, model, DEVICE)

    if debias:

        debias_test(mode[0], 
                    select_layer[0], 
                    model, 
                    experiment_set, 
                    tokenizer,
                    DEVICE, 
                    layers, 
                    heads,
                    counterfactual_paths,
                    label_maps,
                    is_group_by_class, 
                    is_averaged_embeddings, 
                    intervention_type = intervention_type)

    if is_traced:
        
        trace_counterfactual(mode[0], 
                            select_layer[0],
                            model, 
                            save_nie_set_path, 
                            tokenizer,
                            DEVICE, 
                            layers, 
                            heads,
                            counterfactual_paths,
                            label_maps,
                            is_group_by_class, 
                            is_averaged_embeddings, 
                            intervention_type, 
                            debug)

    if is_prediction:
    
        get_predictions(mode[0], 
                        select_layer[0],
                        model,
                        tokenizer,
                        DEVICE, 
                        layers, 
                        heads,
                        counterfactual_paths,
                        label_maps,
                        dev_path,
                        dev_json,
                        is_group_by_class, 
                        is_averaged_embeddings,
                        k=k,
                        num_top_neurons=num_top_neurons,
                        intervention_type=intervention_type)

if __name__ == "__main__":
    main()

