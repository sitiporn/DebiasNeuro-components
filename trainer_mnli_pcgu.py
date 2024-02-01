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
from torch.optim import Adam
# from my_package.utils import  report_gpu
from my_package.cma_utils import collect_counterfactuals, trace_counterfactual, geting_counterfactual_paths, get_single_representation, geting_NIE_paths
from my_package.optimization_utils import test_restore_weight
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from my_package.cma import get_candidate_neurons 
from my_package.data import ExperimentDataset, Dev, get_conditional_inferences, eval_model, print_config
from my_package.optimization_utils import trace_optimized_params, initial_partition_params
from my_package.optimization import partition_param_train, restore_original_weight
from my_package.data import rank_losses
from my_package.intervention import intervene, high_level_intervention
from my_package.cma import cma_analysis,  get_distribution
from my_package.utils import debias_test
from my_package.cma_utils import get_nie_set
import yaml
from my_package.utils import get_num_neurons, get_params, get_diagnosis
from my_package.data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from my_package.optimization import intervene_grad
from transformers import Trainer
# from torch.utils.data import Dataset, DataLoader
from typing import Optional
from my_package.data import get_all_model_paths
from my_package.data import CustomDataset
from my_package.optimization_utils import get_advantaged_samples
from my_package.optimization import partition_param_train, restore_original_weight
from my_package.optimization import CustomAdamW
from my_package.utils import load_model
from my_package.utils import compare_weight
from transformers import AdamW
import torch.optim as optim
import evaluate
import allennlp
from transformers import BertConfig, BertModel
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from trainer_package.custom_trainer import CustomTrainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# def main():
global tokenizer
global label_maps
global metric

parser = argparse.ArgumentParser()
parser.add_argument("--model_save_path", type=str, default=None, required=True, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--model_load_path", type=str, default=None, required=True, help="The directory where the model checkpoints will be read to train.")
parser.add_argument("--dataset_name", type=str, help="dataset name to train") 
parser.add_argument("--method_name", type=str, help="method to train") 
parser.add_argument("--seed", type=int, default=1548, help="The random seed value")	
parser.add_argument("--random_adv", type=bool,  default=False,  help="The random advanated samples")	
parser.add_argument("--collect_adv", type=bool, default=False,  help="collect advanated samples")	
parser.add_argument("--correct_pred", type=bool, default=True, help="model to filter advantaged samples")	
parser.add_argument("--num_epochs", type=int, default=15, help="Total number of training epochs.")
parser.add_argument("--candidated_class", type=str, help="class used to filter advantaged samples") 
parser.add_argument("--intervention_class", type=str, help="class used to compute NIE scores") 
parser.add_argument("--intervention_type", type=str, help="tye of neuron intervention") 
parser.add_argument("--top_neuron_mode", type=str, help="mode select neurons to perform gradients unlearning") 
parser.add_argument("--grad_direction", type=str, help="moving gradient directoin used to learn or unlearn") 
parser.add_argument("--k", type=int, default=None, help="the percentage of total number of top neurons") 
parser.add_argument("--top_neuron_num", type=int, default=None, help="the number of top neurons") 
parser.add_argument("--compare_frozen_weight", type=bool, default=True, help="compare weight to reference model to restore back to model during training at each step")	
parser.add_argument("--is_averaged_embeddings", type=bool, default=True, help="Average representation across samples")	
parser.add_argument("--DEBUG", type=int, default=0, help="Mode used to debug")	

args = parser.parse_args()
print(args)

if args.dataset_name == 'mnli':
    config_path = "./configs/pcgu_config.yaml"
elif args.dataset_name == 'qqp':
    config_path = "./configs/pcgu_config_qqp.yaml"
elif args.dataset_name == 'fever':
    config_path = "./configs/pcgu_config_fever.yaml"

with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(f'{args.dataset_name} config : {config_path}')

output_dir = args.model_save_path
LOAD_MODEL_PATH = args.model_load_path
method_name =  args.method_name
config['seed'] = args.seed 
method_name = args.method_name 
config['dataset_name'] = args.dataset_name
config['random_adv'] = args.random_adv
config['collect_adv'] = args.collect_adv
config['correct_pred'] = args.correct_pred
config['num_epochs'] =  args.num_epochs
config['candidated_class'] =  [args.candidated_class]
config['intervention_class'] =  [args.intervention_class]
config['intervention_type']  = args.intervention_type
config['top_neuron_mode'] = args.top_neuron_mode
config['grad_direction'] = args.grad_direction
config['k'] = args.k
config['top_neuron_num'] = args.top_neuron_num
config['compare_frozen_weight'] = args.compare_frozen_weight
config['is_averaged_embeddings'] = args.is_averaged_embeddings
config["DEBUG"] = args.DEBUG

dataset = {}
tokenized_datasets = {}
NIE_paths = []
mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
config['treatment'] = mode[0]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = config['dataset_name']
metric = evaluate.load(config["validation_metric"])

print(f'current dataset_name: {dataset_name}')
# ******************** PATH ********************
if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
model_path = config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])] 
# prepare validation for computing NIE scores
tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
# path to save NIE scores
NIE_paths, _ = geting_NIE_paths(config, method_name, mode)
print(f'Loading path for single at seed:{config["seed"]}, layer: {config["layer"]}')
print(f'NIE_paths: {NIE_paths}')

# random seed
seed = config['seed'] 
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
else: 
    seed = str(seed)

print(f'current {method_name} : seed : {seed}')


if not os.path.exists(output_dir): os.mkdir(output_dir) 
output_dir = os.path.join(output_dir, "seed_"+ str(seed))
if not os.path.exists(output_dir): os.mkdir(output_dir) 

# *************** Train model stuff ***************
label_maps = config['label_maps'] 
metric = evaluate.load(config["validation_metric"])
model_config = BertConfig(config['model']["model_name"])
model_config.num_labels = len(label_maps.keys())
model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
reference_model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if config['data_loader']['batch_sampler']['dynamic_padding'] else None

# ************************** PCGU ***************************
# NOTE: scale gradient by confident scores <- kind of reweighting (sample base)
# spatial adaptive gradient scaler by NIE scores (NIE base)
hooks = []
advantaged_bias = None
advantaged_main, advantaged_bias = get_advantaged_samples(config, model, seed, metric=metric, LOAD_MODEL_PATH=LOAD_MODEL_PATH, is_load_model=True, method_name=method_name,device=DEVICE,collect=config['collect_adv'])

if config['model']['is_load_trained_model']:
    all_paths = get_all_model_paths(LOAD_MODEL_PATH)
    path = all_paths[str(seed)]
    model = load_model(path=path, model=model, device=DEVICE)
    print(f'Loading updated model from : {path} to optimize on PCGU done!')
    reference_model = load_model(path=path, model=reference_model, device=DEVICE)
    print(f'Loading reference model from : {path} done!')
else:
    print(f'Using original model to optimize on PCGU')


get_candidate_neurons(config, method_name, NIE_paths, treatments=mode, debug=False) 
model = initial_partition_params(config, method_name, model, do=mode[0])
model, hooks = intervene_grad(model, hooks=hooks, method_name=method_name, config=config, DEBUG=args.DEBUG)
compare_weight(updated_model=model, reference_model=reference_model)

for data_mode in ["train_data", "validation_data", "test_data"]:
    print(f'************ {data_mode} ************')
    adv_samples =  advantaged_bias if data_mode  == "train_data" else None
    tokenized_datasets[data_mode] = CustomDataset(config, label_maps=label_maps, data_mode=data_mode, adv_samples=adv_samples)

print(f'Config of dataloader') 
print(f'Group by len : {config["data_loader"]["batch_sampler"]["group_by_length"]}')
print(f"Dynamics padding : {config['data_loader']['batch_sampler']['dynamic_padding']}, {data_collator}")

# ************************** Training ***************************
lr = float(config['optimizer']['lr']) # follow PCGU papers
pcgu_epochs = 15
pcgu_num_batch_per_epoch = 4
pcgu_batch_size = 64
our_samples = len(advantaged_bias) * config['num_epochs']
pcgu_samples = (pcgu_batch_size * pcgu_num_batch_per_epoch * pcgu_epochs)
lr = (pcgu_samples /  our_samples) * lr

print(f'Our Learning samples : {our_samples}')
print(f'Pcgu Learning samples : {pcgu_samples}')
print(f'learning_rate : {lr}')
print(f'Total epochs : {config["num_epochs"]}')
print(f'random advantaged samples: {config["random_adv"]}')
print(f'top neruon mode : {config["top_neuron_mode"]}')
print(f'#advantaged sampled len: {len(advantaged_bias)}')
print(f'Predict candidated class correct : {config["correct_pred"]}')
print(f'intervention type : {config["intervention_type"]}')

training_args = TrainingArguments(output_dir = output_dir,
                                report_to="none",
                                overwrite_output_dir = True,
                                # steps
                                evaluation_strategy=config['evaluation_strategy'],
                                # num of steps
                                eval_steps=config['eval_steps'],
                                learning_rate = float(config['optimizer']['lr']),
                                weight_decay = config['optimizer']['weight_decay'],
                                per_device_train_batch_size = config["data_loader"]["batch_sampler"]["batch_size"],
                                per_device_eval_batch_size=config["data_loader"]["batch_sampler"]["batch_size"],
                                num_train_epochs = config["num_epochs"],
                                seed=seed,
                                load_best_model_at_end=config["load_best_model_at_end"],
                                metric_for_best_model = config['validation_metric'], # used for criterion for best model
                                greater_is_better = True,  #  used for criterion for best model
                                save_total_limit= config["save_total_limit"],
                                half_precision_backend = config["half_precision_backend"],
                                group_by_length = config["data_loader"]["batch_sampler"]["group_by_length"],
                                )


opitmizer = CustomAdamW(config=config,
                        params=model.parameters(),
                        original_model= reference_model,
                        seed = seed,
                        method_name=method_name,
                        DEVICE=DEVICE,
                        collect_param=config['collect_param'],
                        lr= lr , 
                        weight_decay = config['optimizer']['weight_decay'])
# opitmizer = AdamW(params=model.parameters(), lr= lr , weight_decay = config['optimizer']['weight_decay'])

trainer = CustomTrainer(
    model,
    training_args,
    train_dataset= tokenized_datasets["train_data"],
    eval_dataset= tokenized_datasets["validation_data"],
    tokenizer =tokenizer, 
    compute_metrics=compute_metrics,
    optimizers = (opitmizer, None),
    data_collator=data_collator,
    )

trainer.train()

# if __name__ == "__main__":
#     main()
     
