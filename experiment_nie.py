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
from utils import  report_gpu
from cma_utils import collect_counterfactuals, trace_counterfactual, geting_counterfactual_paths, get_single_representation, geting_NIE_paths
from optimization_utils import test_restore_weight
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import numpy as np
from pprint import pprint
from cma import get_candidate_neurons 
#from nn_pruning.patch_coordinator import (
#    SparseTrainingArguments,
#    ModelPatchingCoordinator,
#)
from data import ExperimentDataset, Dev, get_conditional_inferences, eval_model, print_config
from optimization_utils import trace_optimized_params, initial_partition_params
from optimization import partition_param_train, restore_original_weight
from data import rank_losses
from intervention import intervene, high_level_intervention
from cma import cma_analysis,  get_distribution
from utils import debias_test
from cma_utils import get_nie_set_path
import yaml
from utils import get_num_neurons, get_params, get_diagnosis
from data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from optimization import intervene_grad
from transformers import Trainer
# from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import IterableDataset, RandomSampler, Sampler
import evaluate
import allennlp
from typing import Optional
from slanted_triangular import SlantedTriangular
from torch.optim import Adam
from transformers import BertConfig, BertModel
from transformers.optimization import get_scheduler
from transformers.trainer_utils import has_length, ShardedDDPOption
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import DistributedTensorGatherer,  SequentialDistributedSampler
from torch.utils.data.sampler import SequentialSampler            
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput
from transformers.trainer_callback import TrainerCallback
from trainer_pt_utils import CustomLabelSmoother, test_bucket_iterator
from transformers.trainer_utils import denumpify_detensorize
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import logging
from trainer_pt_utils import RandomSampler, SequentialSampler, BucketBatchSampler, BatchSampler, LengthGroupedSampler
import math
import time
from data import CustomDataset
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from torch.utils.data.distributed import DistributedSampler
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import is_torch_tpu_available
from transformers.trainer_utils import speed_metrics, TrainOutput
from data import get_all_model_paths
from optimization_utils import get_advantaged_samples

config_path = "./configs/cma_experiment.yaml"
with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

dataset = {}
tokenized_datasets = {}
NIE_paths = []
mode = ["High-overlap"]  if config['treatment'] else  ["Low-overlap"] 
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = config['dataset_name']
print(f'current dataset_name: {dataset_name}')
# ******************** PATH ********************
# Model to intervene
# LOAD_MODEL_PATH = '../models/recent_baseline/' 
LOAD_MODEL_PATH = '../models/pcgu_posgrad_replace_5k_recent_baseline/'
# LOAD_MODEL_PATH = '../models/poe2/' 
# method_name =  f'separation_{config["intervention_type"]}_intervention_{LOAD_MODEL_PATH.split("/")[-2]}'
method_name =  f'separation_pcgu_posgrad_replace_5k_recent_baseline'

save_nie_set_path = f'../pickles/{dataset_name}_class_level_nie_{config["num_samples"]}_samples.pickle' if config['is_group_by_class'] else f'../pickles/{dataset_name}_nie_{config["num_samples"]}_samples.pickle'
tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])

if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
model_path = config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])] 

# prepare validation for computing NIE scores
experiment_set = ExperimentDataset(config, encode = tokenizer, dataset_name=dataset_name)                            
exp_loader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)

if not os.path.isfile(save_nie_set_path): get_nie_set_path(config, experiment_set, save_nie_set_path)
counterfactual_paths, _ = geting_counterfactual_paths(config, method_name)
# path to save NIE scores
NIE_paths, _ = geting_NIE_paths(config, method_name, mode)
print(f'Loading path for single at seed:{config["seed"]}, layer: {config["layer"]}')
for path in counterfactual_paths: print(f"{sorted(path.split('_'), key=len)[0]}: {path}")
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

label_maps = config['label_maps'] 
model_config = BertConfig(config['model']["model_name"])
model_config.num_labels = len(label_maps.keys())
model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))

assert config["intervention_type"] == 'replace', f'intervention type is not replace mode'
# ******************* Causal Mediation Analysis ********************* 
collect_counterfactuals(model, model_path, dataset_name, method_name, seed, counterfactual_paths, config, experiment_set, exp_loader, tokenizer, DEVICE=DEVICE) 

# Todo: recheck is hooks in the model
cma_analysis(config, 
            model_path,
            method_name, 
            seed, 
            counterfactual_paths, 
            NIE_paths, 
            save_nie_set_path = save_nie_set_path, 
            model = model, 
            treatments = mode, 
            tokenizer = tokenizer, 
            experiment_set = experiment_set, 
            DEVICE = DEVICE, 
            DEBUG = True)   

get_candidate_neurons(config, method_name, NIE_paths, treatments=mode, debug=False, mode=config['top_neuron_mode']) 