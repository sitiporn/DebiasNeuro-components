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
from my_package.data import reweight_nie
from learn_mask.sparse_trainer import SparseTrainer
from learn_mask.hf_model import SequenceClassificationTransformer
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

parser = argparse.ArgumentParser()
parser.add_argument("--model_load_path", type=str, default=None, required=True, help="The directory where the model checkpoints will be read to train.")
parser.add_argument("--model_save_path", type=str, default=None, required=True, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--dataset_name", type=str, help="dataset name to train") 
parser.add_argument("--method_name", type=str, help="method to train") 
parser.add_argument("--mask_scores_learning_rate", type=float, help="learning rate for mask")	
parser.add_argument("--seed", type=int, default=1548, help="The random seed value")	
parser.add_argument("--save_steps", type=int, default=500, help="the number of step to save model")
parser.add_argument("--eval_steps", type=int, default=500, help="the number of step to save model")
parser.add_argument("--num_epochs", type=int, default=15, help="Total number of training epochs.")
parser.add_argument("--DEBUG", type=int, default=0, help="Mode used to debug")	

args = parser.parse_args()
print(args)

output_dir = args.model_save_path
LOAD_MODEL_PATH = args.model_load_path
method_name =  args.method_name

if args.dataset_name == 'mnli':
    config_path = 'configs/learn_mask/train_config_mnli.yaml'
elif args.dataset_name == 'qqp':
    config_path = 'configs/learn_mask/train_config_qqp.yaml'
elif args.dataset_name == 'fever':
    config_path = 'configs/learn_mask/train_config_fever.yaml'
else: 
    raise Exception("Sorry, no given dataset name")

with open(config_path, "r") as yamlfile: config = yaml.load(yamlfile, Loader=yaml.FullLoader)

config_path = 'configs/learn_mask/unstructured_sigmoid.yaml'
with open(config_path, "r") as yamlfile: sparse_args = yaml.load(yamlfile, Loader=yaml.FullLoader)

config['seed'] = args.seed 
method_name = args.method_name 
config['dataset_name'] = args.dataset_name
config['num_epochs'] =  args.num_epochs
config['save_steps'] = args.save_steps
config['eval_steps'] = args.eval_steps
config["DEBUG"] = args.DEBUG
sparse_args["mask_scores_learning_rate"] = args.mask_scores_learning_rate

is_load_model = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
label_maps = config['label_maps']

# random seed
seed = int(config["seed"])
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if not os.path.exists(output_dir): os.mkdir(output_dir) 
output_dir = os.path.join(output_dir, "seed_"+ str(seed))
if not os.path.exists(output_dir): os.mkdir(output_dir) 
# "validation_metric": "+accuracy"?

model_name  =  config['model']['model_name']
metric = evaluate.load(config["validation_metric"])
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SequenceClassificationTransformer.from_pretrained(model_name, num_labels = len(label_maps))

from my_package.data import get_all_model_paths, get_all_checkpoints
all_paths = get_all_model_paths(LOAD_MODEL_PATH)
path = all_paths[str(seed)]

if is_load_model:
    from my_package.utils import load_model
    model = load_model(path=path, model=model, device=device)
    print(f'Load=True: Loading model from : {path}')
else:
    print(f'Using original model')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if config['data_loader']['batch_sampler']['dynamic_padding'] else None
dataset = {}
tokenized_datasets = {}

for data_mode in ["train_data", "validation_data", "test_data"]:
    print(f'************ {data_mode} ************')
    tokenized_datasets[data_mode] = CustomDataset(config, label_maps=label_maps, data_mode=data_mode)

print(f'Config of dataloader') 
print(f'Group by len : {config["data_loader"]["batch_sampler"]["group_by_length"]}')
print(f"Dynamics padding : {config['data_loader']['batch_sampler']['dynamic_padding']}, {data_collator}")
print(f'is_load_model: {is_load_model}')
print(f'seed : {seed}')
print(f"Learning_rate : {float(config['optimizer']['lr'])}")
print(f'mask_scores_learning_rate: {sparse_args["mask_scores_learning_rate"]}')
print(f'#Epochs : {config["num_epochs"]}')
print(f'eval steps : {config["eval_steps"]}')
print(f'save steps : {config["save_steps"]}')
print(f'label maps : {label_maps}')

# Todo:
# remove magic number: number of max steps in sparse trainer 
training_args = TrainingArguments(output_dir = output_dir,
                                report_to="none",
                                overwrite_output_dir = True,
                                # steps
                                evaluation_strategy=config['evaluation_strategy'],
                                # num of steps
                                eval_steps=config['eval_steps'],
                                save_steps=config['save_steps'],
                                learning_rate = float(config['optimizer']['lr']),
                                weight_decay = config['optimizer']['weight_decay'],
                                warmup_ratio = config['warmup_ratio'], 
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

trainer = SparseTrainer(
    model,
    training_args,
    sparse_args=SparseTrainingArguments(**sparse_args),
    freeze_weights=True,
    train_dataset= tokenized_datasets["train_data"],
    eval_dataset= tokenized_datasets["validation_data"],
    tokenizer =tokenizer, 
    compute_metrics=compute_metrics,
    optimizers = (None, None),
    data_collator=data_collator,
    )

trainer.train()

# compile model here from all checkpoints
# checkpoints = get_all_checkpoints(trainer.args.output_dir)
# for  checkpoint_path in checkpoints: 
#     print(f'checkpoint path : {checkpoint_path}')
#     cur_model = trainer.compile_model(checkpoint_path)
#     torch.save(cur_model, checkpoint_path)

