import sys
import pickle
from tqdm import tqdm
import os 
import yaml
import torch
import numpy as np
from utils import get_ans
import operator
from data import get_result, get_num_neurons, get_params

# ++++++  config ++++++++++++

with open("config.yaml", "r") as yamlfile:
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

if config['get_result']: get_result(config, eval_path, prediction_path, neuron_path, top_neuron, prediction_mode, params, digits)

num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

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

        result_path = os.path.join(os.path.join(eval_path, epsilon_path),  result_path)

        # cur_num_neurons = result_path.split("/")[-1].split('_')[5].split('-')[0]
        # cur_eps = result_path.split('/')[3].split('v')[-1]

        with open(result_path, 'rb') as handle: 

            current_score = pickle.load(handle)

        cur_score = []

        for type in ['entailed','non-entailed']:

            class_score = []

            for score in ['lexical_overlap', 'subsequence','constituent']:

                class_score.append(current_score[group][type][score])

            cur_score.append(class_score)

        cur_score = torch.mean(torch.mean(torch.Tensor(cur_score), dim=-1),dim=0)
        scores[f"{epsilon}-{group}"] = cur_score
        
# key_rank_scores = list(rank_scores.keys())
# best_score_key = list(rank_scores.keys())[0]
# null_score_key = list(rank_scores.keys())[48]

# print(f"+++++++++++++ Config +++++++++++++++++++")
# print(f"Low: {low} , High : {high}, Step : {step}")
# print(f"optimize intervention scores : {rank_scores[best_score_key]} on weaken rate at {best_score_key.split('-')[0]}, {best_score_key.split('-')[1]} neurons")
# print(f"Null scores : {rank_scores[null_score_key]} with {null_score_key.split('-')[-1]} neurons")

# # # Todo: grid search on X percents and weaken rates


# if config['save_rank']:

#     with open('../pickles/evaluations/ranks/combine_scores.pickle', 'wb') as handle: 
#         pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         print(f"saving scores object")
