import os
import os.path
import re
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
from my_package.utils import  LayerParams
from my_package.cma_utils import get_overlap_thresholds, group_by_treatment, get_hidden_representations
from my_package.intervention import Intervention, neuron_intervention, get_mediators
from my_package.utils import get_ans, compute_acc, compute_maf1
from my_package.utils import get_num_neurons, get_params, relabel, give_weight
from torch.optim import Adam
from transformers import AutoTokenizer, BertForSequenceClassification
from functools import partial
from my_package.utils import load_model
from my_package.utils import compare_frozen_weight, prunning_biased_neurons
from my_package.get_bias_samples_fever import get_ngram_doc, get_ngram_docs, vanilla_tokenize
from my_package.counter import count_negations 
from my_package.data import get_all_model_paths, Dev, DevFever, DevQQP, get_hans_result


def eval_fairness_mnli(model, config, tokenizer, DEVICE, LOAD_MODEL_PATH, method_name, is_load_model=True, is_optimized_set = False):
    """ to get predictions and score on test and challenge sets"""
    distributions = {}
    losses = {}
    golden_answers = {}

    
    all_paths = get_all_model_paths(LOAD_MODEL_PATH)
    OPTIMIZED_SET_JSONL = config['dev_json']
    # datasets
    VALIDATION_SET_JSONL = 'multinli_1.0_dev_matched.jsonl'
    IN_DISTRIBUTION_SET_JSONL = 'multinli_1.0_dev_mismatched.jsonl'
    CHALLENGE_SET_JSONL = 'heuristics_evaluation_set.jsonl' 
    RESULT_PATH = f'../pickles/fairness_performances/'
    if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)
    json_sets = [OPTIMIZED_SET_JSONL] if is_optimized_set else [VALIDATION_SET_JSONL, IN_DISTRIBUTION_SET_JSONL, CHALLENGE_SET_JSONL]
    # valid and test set
    acc_avg = {}
    entail_avg = {}
    contradiction_avg = {}
    neutral_avg = {}
    computed_acc_count = {}
    # challenge set  
    hans_avg = 0
    entail_hans_avg = 0
    non_entail_hans_avg = 0
    computed_hans_count = 0
    # collect 
    acc_in = {}
    acc_out = [] 

    for cur_json in json_sets:
        name_set = list(cur_json.keys())[0] if is_optimized_set else cur_json.split("_")[0] 
        if name_set == 'multinli': name_set = name_set + '_' + cur_json.split("_")[-1].split('.')[0]
        if 'multinli' not in name_set: continue
        acc_avg[name_set] = 0
        entail_avg[name_set] = 0
        contradiction_avg[name_set] = 0
        neutral_avg[name_set] = 0
        computed_acc_count[name_set] = 0
        acc_in[name_set] = []
    
    seed = str(config['seed'])
    if not config['compute_all_seeds']:  all_paths = {seed: all_paths[seed]}
    
    for seed, path in all_paths.items():
        if is_load_model:
            model = load_model(path=path, model=model)
        else:
            print(f'Using original model')
        
        for cur_json in json_sets:
            name_set = list(cur_json.keys())[0] if is_optimized_set else cur_json.split("_")[0] 
            if name_set == 'multinli': name_set = name_set + '_' + cur_json.split("_")[-1].split('.')[0]
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

                acc_in[name_set].append(acc['all'])    
                print(f"overall acc : {acc['all']}")
                print(f"contradiction acc : {acc['contradiction']}")
                print(f"entailment acc : {acc['entailment']}")
                print(f"neutral acc : {acc['neutral']}")
                print(f"|En - Cont| : {abs(acc['entailment'] - acc['contradiction'])}")
                print(f"|En - Neu| : {abs(acc['entailment'] - acc['neutral'])}")
                # score_diff =  ( abs(ent - cont) + abs(ent-neu) ) / 2
                score_diff =  (abs(acc['entailment'] - acc['contradiction']) +  abs(acc['entailment'] - acc['neutral']) ) / 2
                print(f'score_diff: {score_diff}')

                acc_avg[name_set] += acc['all']
                entail_avg[name_set] += acc['entailment']
                neutral_avg[name_set] += acc['neutral']
                contradiction_avg[name_set] += acc['contradiction']
                computed_acc_count[name_set] += 1

            elif config['get_hans_result'] and 'heuristics'in cur_json: 
                cur_hans_score, entail_score, non_entail_score = get_hans_result(cur_raw_distribution_path, config)
                hans_avg += cur_hans_score
                entail_hans_avg += entail_score
                non_entail_hans_avg += non_entail_score
                computed_hans_count +=1
                print(f'has score :{cur_hans_score}')
                print(f'Entail score: {entail_score}')
                print(f'Non-entail: {non_entail_score}')
                print(f'diff_score : {abs(entail_score-non_entail_score)}')
                acc_out.append(cur_hans_score.item())
    
    print(f'==================== Avearge scores ===================')

    # 1.MNLI: 
    #   |Entailment - Contradiction|, |Entailment -Neutral|  แล้วเอามาเฉีล่ย
    #   1.1 HANS   |Entailment-Nonentailment|
    # 2. FEVER: support vs nonsupport

    for name_set in acc_in.keys():
        print(f"average {name_set} overall acc : {acc_avg[name_set] / len(all_paths)}")
        cont = contradiction_avg[name_set] / len(all_paths)
        ent = entail_avg[name_set] / len(all_paths)
        neu = neutral_avg[name_set] /  len(all_paths)
        print(f'Contradiction : {cont}')
        print(f'Entailment: {ent}')
        print(f'Neutral: {neu}')
        score_diff =  ( abs(ent - cont) + abs(ent-neu) ) / 2
        print(f'Score_diff :{score_diff}')
  
    ent = entail_hans_avg / len(all_paths)
    non = non_entail_hans_avg / len(all_paths)
    hans_diff = abs(ent - non) 
    print(f'avarge hans score : { hans_avg  / len(all_paths)}')
    print(f'Hans_diff : {hans_diff}')

def eval_fairness_qqp(model, config, tokenizer, DEVICE, LOAD_MODEL_PATH, is_load_model=True, is_optimized_set = False):
    """ to get predictions and score on test and challenge sets"""
    distributions = {}
    losses = {}
    golden_answers = {}
    
    all_paths = get_all_model_paths(LOAD_MODEL_PATH)
    OPTIMIZED_SET_JSONL = config['dev_json']
    # datasets
    IN_DISTRIBUTION_SET_JSONL = 'qqp.dev.jsonl'
    CHALLENGE_SET_JSONL = 'paws.dev_and_test.jsonl' 
    RESULT_PATH = f'../pickles/performances/'
    json_sets = [OPTIMIZED_SET_JSONL] if is_optimized_set else [IN_DISTRIBUTION_SET_JSONL, CHALLENGE_SET_JSONL]
    maf1_avg = 0
    is_dup_avg = 0
    not_dup_avg = 0   
    paws_is_dup_avg = 0
    paws_not_dup_avg = 0   
    paws_avg = 0
    computed_maf1_count = 0
    computed_paws_count = 0
    acc_in = []
    acc_out = []
    
    for seed, path in all_paths.items():
        if is_load_model:
            from my_package.utils import load_model
            model = load_model(path=path, model=model,device=DEVICE)
        else:
            print(f'Using original model')
        for cur_json in json_sets:
            name_set = list(cur_json.keys())[0] if is_optimized_set else cur_json.split("_")[0] 
            distributions = []
            losses = []
            golden_answers = []
            data_name = "test_data"
            
            dev_set = DevQQP(config['dev_path'] , cur_json)
            dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)

            # tokenized_datasets = {}
            # tokenized_datasets[data_name] = FeverDatasetClaimOnly(config, label_maps=config['label_maps'])
            # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
            # def collator(input_samples):
            #     x = data_collator(input_samples)
            #     for i in x:
            #         i.pop("claim",None)
            #     return x
            
            # dev_loader = DataLoader(tokenized_datasets[data_name], batch_size = 32, shuffle = False,  collate_fn=collator)
            model.eval()
            for batch_idx, (inputs) in enumerate( t := tqdm(dev_loader)):
                
                cur_inputs = {} 
                t.set_description(f'{name_set} batch_idx {batch_idx}/{len(dev_loader)}')
                for idx, (cur_inp, cur_col) in enumerate(zip(inputs, list(dev_set.df.keys()))): cur_inputs[cur_col] = cur_inp
                # get the inputs 
                pair_sentences = [[sentence1, sentence2] for sentence1, sentence2 in zip(cur_inputs['sentence1'], cur_inputs['sentence2'])]
                pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
                pair_sentences = {k: v.to(DEVICE) for k,v in pair_sentences.items()}
                label_ids = torch.tensor([label for label in cur_inputs['label']]) 
                if label_ids is not None: label_ids = label_ids.to(DEVICE)

                with torch.no_grad(): 
                    # Todo: generalize to distribution if the storage is enough
                    outs =  model(**pair_sentences, labels= label_ids if  'heuristics' not in cur_json else None)
                    distributions.extend(F.softmax(outs.logits.cpu() , dim=-1))
                    golden_answers.extend(label_ids.cpu() if label_ids is not None else cur_inputs['label'])

            cur_raw_distribution_path = os.path.join(RESULT_PATH, f'inference_{name_set}.pickle')
            
            with open(cur_raw_distribution_path, 'wb') as handle: 
                pickle.dump(distributions , handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(golden_answers, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(losses[cur_json.split("_")[0]], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving without condition distribution into : {cur_raw_distribution_path}')
            
            if 'paws' not in cur_json: 
                maf1 = compute_maf1(cur_raw_distribution_path, config["label_maps"])
               
                if 'Null' in maf1.keys():
                    maf1 = maf1['Null']
                print(f"overall maf1 : {maf1['all']}")
                print(f"is_duplicate maf1 : {maf1['duplicate']}")
                print(f"not_duplicate maf1 : {maf1['not_duplicate']}")
                # print(f"neutral acc : {acc['neutral']}")
                maf1_avg += maf1['all']
                acc_in.append(maf1['all'])
                is_dup_avg += maf1['duplicate']
                not_dup_avg += maf1['not_duplicate']
                computed_maf1_count += 1
            elif config['get_paws_result'] and 'paws'in cur_json: 
                maf1_paws = compute_maf1(cur_raw_distribution_path, config["label_maps"])
               
                if 'Null' in maf1_paws.keys():
                    maf1_paws = maf1_paws['Null']
                print(f"overall paws maf1 : {maf1_paws['all']}")
                print(f"not_dup paws maf1 : {maf1_paws['not_duplicate']}")
                print(f"is_dup paws maf1 : {maf1_paws['duplicate']}")
                # print(f"neutral acc : {acc['neutral']}")

                paws_avg += maf1_paws['all']
                acc_out.append(maf1_paws['all'])
                paws_not_dup_avg += maf1_paws['not_duplicate']
                paws_is_dup_avg += maf1_paws['duplicate']
                computed_paws_count += 1
                # breakpoint()
                # cur_hans_score = get_symm_result(cur_raw_distribution_path, config)
                # symm_avg += cur_hans_score
                # print(f'symm score :{cur_hans_score}')
    
    print(f'==================== Average scores ===================')
    print(f">> average overall maf1 : {maf1_avg / len(all_paths)}")
    not_dup_avg = not_dup_avg / len(all_paths)
    is_dup_avg = is_dup_avg  / len(all_paths)
    print(f"averge not dup maf1 : {not_dup_avg}")
    print(f"average is dup maf1 : {is_dup_avg }")
    print(f'>>diff_score: {abs(not_dup_avg - is_dup_avg)}')
                              
    print(f'>>avarge paws score : { paws_avg  / len(all_paths)}')
    paws_not_dup_avg = paws_not_dup_avg / len(all_paths)
    paws_is_dup_avg  = paws_is_dup_avg   / len(all_paths)
    print(f"averge paws not dup maf1 : {paws_not_dup_avg}")
    print(f"average paws dup SUPPORTS maf1 : {paws_is_dup_avg}") 
    print(f'>>diff_score: {abs(paws_not_dup_avg - paws_is_dup_avg)}')


def eval_fairness_fever(model, config, tokenizer, DEVICE, LOAD_MODEL_PATH, is_load_model=True, is_optimized_set = False):
    """ to get predictions and score on test and challenge sets"""
    distributions = {}
    losses = {}
    golden_answers = {}

    
    all_paths = get_all_model_paths(LOAD_MODEL_PATH)
    OPTIMIZED_SET_JSONL = config['dev_json']
    # datasets
    IN_DISTRIBUTION_SET_JSONL = 'fever.dev.jsonl'
    CHALLENGE_SET1_JSONL = 'fever_symmetric_v0.1.test.jsonl' 
    CHALLENGE_SET2_JSONL = 'fever_symmetric_v0.2.test.jsonl' 
    RESULT_PATH = f'../pickles/performances/'
    json_sets = [OPTIMIZED_SET_JSONL] if is_optimized_set else [IN_DISTRIBUTION_SET_JSONL, CHALLENGE_SET1_JSONL,CHALLENGE_SET2_JSONL]
    # json_sets = [OPTIMIZED_SET_JSONL] if is_optimized_set else [CHALLENGE_SET2_JSONL]
    # json_sets = [OPTIMIZED_SET_JSONL] if is_optimized_set else [IN_DISTRIBUTION_SET_JSONL, CHALLENGE_SET1_JSONL]
    acc_avg = 0
    support_avg = 0
    refute_avg = 0
    notenough_avg = 0

    symm1_support_avg = 0
    symm1_refute_avg = 0
    symm1_notenough_avg = 0
    symm1_avg = 0

    symm2_support_avg = 0
    symm2_refute_avg = 0
    symm2_notenough_avg = 0
    symm2_avg = 0

    computed_acc_count = 0
    computed_symm1_count = 0
    computed_symm2_count = 0

    acc_in = []
    symm1_out =  []
    symm2_out =  []

    for seed, path in all_paths.items():
        if is_load_model:
            from my_package.utils import load_model
            model = load_model(path=path, model=model,device=DEVICE)
        else:
            print(f'Using original model')
        for cur_json in json_sets:
            name_set = list(cur_json.keys())[0] if is_optimized_set else cur_json.split("_")[0] 
            distributions = []
            losses = []
            golden_answers = []
            data_name = "test_data"
            dev_set = DevFever(config['dev_path'] , cur_json)
            dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)
            # tokenized_datasets = {}
            # tokenized_datasets[data_name] = FeverDatasetClaimOnly(config, label_maps=config['label_maps'])
            # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
            # def collator(input_samples):
            #     x = data_collator(input_samples)
            #     for i in x:
            #         i.pop("claim",None)
            #     return x
            
            # dev_loader = DataLoader(tokenized_datasets[data_name], batch_size = 32, shuffle = False,  collate_fn=collator)
            model.eval()
            for batch_idx, (inputs) in enumerate( t := tqdm(dev_loader)):
                
                cur_inputs = {} 
                t.set_description(f'{name_set} batch_idx {batch_idx}/{len(dev_loader)}')
                for idx, (cur_inp, cur_col) in enumerate(zip(inputs, list(dev_set.df.keys()))): cur_inputs[cur_col] = cur_inp
                # get the inputs 
              
                # get the inputs 
                if config['claim_only']:
                    pair_sentences = [sentence1 for sentence1 in cur_inputs['claim']]       
                    pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
                    pair_sentences = {k: v.to(DEVICE) for k,v in pair_sentences.items()}
                # ignore label_ids when running experiment on hans
                else:
                    pair_sentences = [[sentence1, sentence2] for sentence1, sentence2 in zip(cur_inputs['evidence_sentence'], cur_inputs['claim'])]
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
            
            if 'symmetric' not in cur_json: 
                acc = compute_acc(cur_raw_distribution_path, config["label_maps"])
               
                if 'Null' in acc.keys():
                    acc = acc['Null']
                print(f"overall acc : {acc['all']}")
                print(f"REFUTES acc : {acc['REFUTES']}")
                print(f"SUPPORTS acc : {acc['SUPPORTS']}")
                acc_in.append(acc['all'])
                # print(f"neutral acc : {acc['neutral']}")

                acc_avg += acc['all']
                refute_avg += acc['REFUTES']
                support_avg += acc['SUPPORTS']
                computed_acc_count += 1
            elif config['get_symm_result'] and 'symmetric_v0.1'in cur_json: 
                acc_symm = compute_acc(cur_raw_distribution_path, config["label_maps"])
               
                if 'Null' in acc_symm.keys():
                    acc_symm = acc_symm['Null']
                print(f"overall symm1 acc : {acc_symm['all']}")
                print(f"REFUTES symm1 acc : {acc_symm['REFUTES']}")
                print(f"SUPPORTS symm1 acc : {acc_symm['SUPPORTS']}")
                # print(f"neutral acc : {acc['neutral']}")
                symm1_out.append(acc_symm['all'])

                symm1_avg += acc_symm['all']
                symm1_refute_avg += acc_symm['REFUTES']
                symm1_support_avg += acc_symm['SUPPORTS']
                computed_symm1_count += 1

            elif config['get_symm_result'] and 'symmetric_v0.2'in cur_json: 
                acc_symm = compute_acc(cur_raw_distribution_path, config["label_maps"])
               
                if 'Null' in acc_symm.keys():
                    acc_symm = acc_symm['Null']
                print(f"overall symm2 acc : {acc_symm['all']}")
                print(f"REFUTES symm2 acc : {acc_symm['REFUTES']}")
                print(f"SUPPORTS symm2 acc : {acc_symm['SUPPORTS']}")
                # print(f"neutral acc : {acc['neutral']}")
                symm2_out.append(acc_symm['all'])

                symm2_avg += acc_symm['all']
                symm2_refute_avg += acc_symm['REFUTES']
                symm2_support_avg += acc_symm['SUPPORTS']
                computed_symm2_count += 1
    
    print(f'==================== Average scores ===================')
    print(f">>average overall acc : {acc_avg / len(all_paths)}")
    refute_avg  = refute_avg / len(all_paths)
    support_avg = support_avg   / len(all_paths)
    print(f"averge REFUTES acc : {refute_avg}")
    print(f"average SUPPORTS acc : {support_avg}")
    print(f'>>score_diff: {abs(refute_avg - support_avg)}')
    # print(acc_in)
    # print(acc_avg)
    
    print(f'>>avarge symm1 score : { symm1_avg  / len(all_paths)}')
    symm1_refute_avg   =  symm1_refute_avg  / len(all_paths)
    symm1_support_avg  =  symm1_support_avg / len(all_paths)
    print(f"averge symm1 REFUTES acc :   {symm1_refute_avg}")
    print(f"average symm1 SUPPORTS acc : {symm1_support_avg}") 
    print(f'>>score_diff: {abs(symm1_refute_avg - symm1_support_avg)}')
    # print(symm1_avg)
    # print(symm1_out)
    
    print(f'>>avarge symm2 score : { symm2_avg  / len(all_paths)}')
    symm2_refute_avg   =  symm2_refute_avg / len(all_paths)
    symm2_support_avg  =  symm2_support_avg / len(all_paths)    
    print(f"averge symm2 REFUTES acc :   {symm2_refute_avg}")
    print(f"average symm2 SUPPORTS acc : {symm2_support_avg}") 
    print(f'>>score_diff: {abs(symm2_refute_avg - symm2_support_avg)}')
    # print(symm2_avg)
    # print(symm2_out)