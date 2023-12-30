import sys
import pickle
from tqdm import tqdm
import os 
import yaml
import torch
import numpy as np
from my_package.utils import get_ans
import operator
from my_package.data import get_condition_inference_hans_result, get_num_neurons, get_params

# ++++++  config ++++++++++++

with open("./configs/experiment_config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(config)

# use to read prediction txt files
# there are three modes (percent, k, neurons)
# percent; custom ranges of percent to search 
# k; specify percents to search
# neurons; the range group of neuron from to neurons
# top_k_mode =  'percent' if config['k'] is not None  else 'neurons' 
top_mode =  'percent' if config['range_percents'] else ('k' if config['k'] else 'neurons')

prediction_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'

eval_path = f'../pickles/evaluations/'
prediction_path = '../pickles/prediction/' 
neuron_path = f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_all_layers.pickle' if config['eval']['all_layers'] else f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_{config["layer"]}.pickle'

with open(neuron_path, 'rb') as handle: 
    top_neuron = pickle.load(handle)

params, digits = get_params(config)

if config['get_condition_hans_result']: get_condition_inference_hans_result(config, eval_path, prediction_path, neuron_path, top_neuron, prediction_mode, params, digits)

num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

# ../pickles/prediction/v0.9/raw_distribution_0.9_High-overlap_all_layers_0.05-k_weaken_hans.pickle

res = {}
scores = {}

rank_scores = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
total_neurons = get_num_neurons(config)

for epsilon in (t := tqdm(params['epsilons'])):  

    epsilon_path = f'v{round(epsilon, digits["epsilons"])}'

    t.set_description(f"epsilon : {epsilon} ")

    for group in params['percent']:

        result_path = f'result_{prediction_mode}_{config["eval"]["intervention_mode"]}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

        if config['eval']['all_layers']: result_path = f'result_{prediction_mode}_{config["eval"]["intervention_mode"]}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

        # result_path = os.path.join(os.path.join(eval_path, epsilon_path),  result_path)
        result_path = '../pickles/evaluations/v0.9/result_0.9_Intervene_all_layers_0.05-k_High-overlap_weaken_hans.txt' 

        # cur_num_neurons = result_path.split("/")[-1].split('_')[5].split('-')[0]
        # cur_eps = result_path.split('/')[3].split('v')[-1]
        # ../pickles/evaluations/v0.9/result_0.9_Intervene_all_layers_0.05-k_High-overlap_weaken_hans.txt
        # result_path : '../pickles/evaluations/v0.9/result_0.9_Intervene_all_layers_0.9-k_High-overlap_weaken_hans.txt'
        # breakpoint()

        with open(result_path, 'rb') as handle: 

            current_score = pickle.load(handle)

        cur_score = []

        for type in ['entailed','non-entailed']:

            class_score = []

            for score in ['lexical_overlap', 'subsequence','constituent']: class_score.append(current_score[group][type][score])

            cur_score.append(class_score)

        cur_score = torch.mean(torch.mean(torch.Tensor(cur_score), dim=-1),dim=0)
        scores[f"{epsilon}-{group}"] = cur_score

        
rank_scores = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))

key_rank_scores = list(rank_scores.keys())
best_score_key = list(rank_scores.keys())[0]
# null_score_key = list(rank_scores.keys())[48]

print(f"+++++++++++++ Config +++++++++++++++++++")
print(f"Low: {config['epsilons']['low']} , High : {config['epsilons']['high']}, Step : {config['epsilons']['step']}")
print(f"weaken rate at {best_score_key.split('-')[0]}")
print(f"masking neuron rate {float(best_score_key.split('-')[1]) * 100 } percent from entire model")
print(f"optimize intervention scores on hans : {rank_scores[best_score_key]}")
# print(f"without  intervention scores on hans : {rank_scores['0.0-0.0']}")

# print(f"Null scores : {rank_scores[null_score_key]} with {null_score_key.split('-')[-1]} neurons")



# # # Todo: grid search on X percents and weaken rates


# if config['save_rank']:

#     with open('../pickles/evaluations/ranks/combine_scores.pickle', 'wb') as handle: 
#         pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"saving scores object")
