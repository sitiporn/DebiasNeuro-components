import sys
import pickle
from tqdm import tqdm
import os 
import yaml
import torch
import numpy as np
from utils import get_ans
import operator
from utils import get_num_neurons, get_params
from data import get_result

# ++++++  config ++++++++++++

with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(config)

# use to read prediction txt files
top_mode =  'percent' if config['k'] is not None  else 'neurons'
prediction_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'

eval_path = f'../pickles/evaluations/'
prediction_path = '../pickles/prediction/' 
neuron_path = f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_all_layers.pickle' if config['eval']['all_layers'] else f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_{config["layer"]}.pickle'

with open(neuron_path, 'rb') as handle: 
    top_neuron = pickle.load(handle)

params, digits = get_params(config)

if config['get_result']: get_result(config, params, eval_path, prediction_path, neuron_path, top_neuron, digits, prediction_mode)

# num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

# res = {}
# scores = {}

# percents = [round(val, digits)for val in np.arange(low, high, step).tolist()]
# digits = len(str(step).split('.')[-1])
# rank_scores = dict(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
# total_neurons = get_num_neurons(config)


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
