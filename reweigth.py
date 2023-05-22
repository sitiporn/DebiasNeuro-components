import os
import os.path
import pandas as pd
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml
from data import ExperimentDataset, Dev, get_predictions, print_config


with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

dev_path = "../debias_fork_clean/debias_nlu_clean/data/nli/"

file = 'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl'

dev_path = os.path.join(os.path.join(dev_path, file))

tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(config["model_name"])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

avg_losses = []

reweighting_set = Dev(config['dev_path'], config['dev_json'])
reweighting_loader = DataLoader(reweighting_set, batch_size = 32, shuffle = False, num_workers=0)
    
mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 

if config['get_prediction']: get_predictions(config, mode[0], model, tokenizer, DEVICE)

# quantifying by using losses
# for index, row in df.iterrows():

#     gold_label = row['gold_label'] 
#     bias_prob = row['bias_probs']
#     weight_score = row['weight_score']

#     print(f"gold_label : {gold_label}, bias_prob : {bias_prob}, weight score : {weight_score}")
    
    # using total loss to quantifying the impact 
    # 1. reweight probs from bias model
    # 2. aggregate with total loss
    # 3. using loss to do hyper parameter search for weaken rate with fixed masking rates
    # where is loss ? 
    # focal loss ? 
    # weight * cross entropy loss; weighted term aggregrate with cross entropy loss of main model
    # breakpoint()

    # step bias only model
    # 1. preparing handcarf feature (bias feature)
    # 2. training bias model 
    # 3. using bias model to quanifying the importance of each instance

    # bias-only model; debias technique by reweight of the importance of instances towards losses of main model once traing mode in competiive methods
    # mitigate paper dont reweighting just training bias-only model using as counterfactual 




# Todo: get label maps 
# check from prediction whether correct or not 
# incorrect; ex 1/0.3 
#   low prob -> high weight 
# correct 1/0.999 -> 1
#   high prob -> low weight    


"""
this dev file that we use 
dev_json = new_dev_df.to_json(orient='records', lines=True)    

with open('dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl', 'w') as json_file:
    json_file.write(dev_json)
dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl
---
using this as metric to quantifying by

1. main model correct, bias model correct -> low loss
2. main model 

bias model -> correct   -> low weight
bias model -> incorrect -> high weight 

reweight probs ;  only the class belonging to the golden answers

"""
