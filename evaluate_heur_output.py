import sys
import pickle
from tqdm import tqdm
import os 
import yaml
import torch
import numpy as np
from utils import get_ans
from data import convert_to_text_ans

def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"
# ++++++  config ++++++++++++

with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(config)

# use to read prediction txt files
top_mode =  'percent' if config['k'] is not None  else 'neurons'
prediction_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'

eval_path = f'../pickles/evaluations/topk_{config["eval"]["do"]}_all_layers_{config["intervention_type"]}.pickle'if config['eval']['all_layers'] else f'../pickles/evaluations/topk_{config["eval"]["do"]}_L{config["layer"]}_{config["intervention_type"]}.pickle'
neuron_path = f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_all_layers.pickle' if config['eval']['all_layers'] else f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_{config["layer"]}.pickle'

convert_to_text_ans(config, neuron_path)
    
