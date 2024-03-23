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
from my_package.utils import  report_gpu
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
#from nn_pruning.patch_coordinator import (
#    SparseTrainingArguments,
#    ModelPatchingCoordinator,
#)
from my_package.data import ExperimentDataset, Dev, get_conditional_inferences, eval_model, print_config, eval_model_qqp
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
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import IterableDataset, RandomSampler, Sampler
import evaluate
import allennlp
from typing import Optional
from trainer_package.slanted_triangular import SlantedTriangular
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
from trainer_package.trainer_pt_utils import CustomLabelSmoother, test_bucket_iterator
from transformers.trainer_utils import denumpify_detensorize
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import logging
from trainer_package.trainer_pt_utils import RandomSampler, SequentialSampler, BucketBatchSampler, BatchSampler, LengthGroupedSampler
import math
import time
from my_package.data import CustomDataset
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
from my_package.data import get_all_model_paths
from my_package.optimization_utils import get_advantaged_samples
from my_package.cma import evalutate_counterfactual


parser = argparse.ArgumentParser()
parser.add_argument("--eval_counterfactuals", type=bool,  default=False)	
parser.add_argument("--collect_counterfactuals", type=bool,  default=False)	
parser.add_argument("--compile_model", type=bool,  default=False)	
parser.add_argument("--compute_nie_scores", type=bool, default=False,  help="collect advanated samples")	
parser.add_argument("--get_candidate_neurons", type=bool, default=False, help="get top neurons looking globally")	
parser.add_argument("--get_top_neurons_layer_each", type=bool, default=False, help="get top neurons looking locally")	
parser.add_argument("--get_top_seq_candidate_neurons", type=bool, default=False, help="get top neurons looking locally")	
parser.add_argument("--intervention_type", type=str, help="tye of neuron intervention") 
parser.add_argument("--intervention_class", type=str, help="class used to compute NIE scores") 
parser.add_argument("--candidated_class", type=str, help="class used to filter advantaged samples") 
parser.add_argument("--is_averaged_embeddings", type=bool, default=False, help="Average counterfactual representation across samples")	
parser.add_argument("--is_group_by_class", type=bool, default=False, help="Grouping counterfactual representation by class")	
parser.add_argument("--dataset_name", type=str, help="dataset to intervene") 
parser.add_argument("--seed", type=int, required=True, help="The random seed value")	
parser.add_argument("--method_name", type=str, help="method of model to intervene") 
parser.add_argument("--model_load_path", type=str, default=None, required=True, help="The directory where the model checkpoints will be read to train.")
parser.add_argument('--treatment', type=str, choices=['High-overlap', 'Low-overlap'], required=True, help="The type of treatment to use as counterfactuals")
parser.add_argument("--k", type=int, default=5, help="the percentage of total number of neurons") 
parser.add_argument("--top_neuron_num", type=int, default=None, help="the number of top neurons") 
parser.add_argument("--top_neuron_layer", type=int, default=None, help="a layer used to optimize") 
parser.add_argument("--DEBUG", type=int, default=0, help="Mode used to debug")	

args = parser.parse_args()
print(args)

if args.dataset_name == 'mnli':
    config_path = "./configs/cma_experiment.yaml"
elif args.dataset_name == 'qqp':
    config_path = "./configs/cma_qqp_experiment.yaml"
elif args.dataset_name == 'fever':
    # Todo: create new config files 
    config_path = "./configs/pcgu_config_fever.yaml"

with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(f'{args.dataset_name} config : {config_path}')

config["eval_counterfactuals"] = args.eval_counterfactuals
config["collect_counterfactuals"] = args.collect_counterfactuals
config["compute_nie_scores"] = args.compute_nie_scores
config["get_candidate_neurons"] = args.get_candidate_neurons 
config["get_top_neurons_layer_each"] = args.get_top_neurons_layer_each
config["intervention_type"] = args.intervention_type
config["intervention_class"] = [args.intervention_class]
config["candidated_class"] = args.candidated_class
config["is_averaged_embeddings"]  = args.is_averaged_embeddings
config["is_group_by_class"] = args.is_group_by_class
config['dataset_name'] = args.dataset_name
config["seed"] = args.seed
config['method_name'] = args.method_name
config["model_load_path"] = args.model_load_path
config["treatment"] = args.treatment
config["DEBUG"] = args.DEBUG
config['k'] = args.k
config['top_neuron_num'] = args.top_neuron_num
config['top_neuron_layer'] = args.top_neuron_layer
config['compile_model'] = args.compile_model


dataset = {}
tokenized_datasets = {}
NIE_paths = []
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_name = config['dataset_name']
method_name = config["method_name"]
treatments = [args.treatment]
print(f'current dataset_name: {dataset_name}')
LOAD_MODEL_PATH = args.model_load_path
# ******************** PATH ********************
if config['is_group_by_class']:
    save_nie_set_path = f'../pickles/{dataset_name}_class_level_nie_{config["num_samples"]}_samples.pickle' 
elif config["is_averaged_embeddings"]:
    save_nie_set_path = f'../pickles/{dataset_name}_nie_{config["num_samples"]}_samples.pickle'

tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
# if os.path.exists(LOAD_MODEL_PATH): all_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
all_model_paths = get_all_model_paths(LOAD_MODEL_PATH, 'compile_model') if config['compile_model'] else get_all_model_paths(LOAD_MODEL_PATH)
model_path = config['seed'] if config['seed'] is None else all_model_paths[str(config['seed'])] 

# prepare validation for Counterfactual Generations
experiment_set = ExperimentDataset(config, encode = tokenizer, dataset_name=dataset_name, DEBUG=config['DEBUG'])                            
exp_loader = DataLoader(experiment_set, batch_size = 32, shuffle = False, num_workers=0)
# not used gettiem of ExperimentDataset
if not os.path.isfile(save_nie_set_path): get_nie_set(config, experiment_set, save_nie_set_path)
counterfactual_paths, _ = geting_counterfactual_paths(config, method_name)
# path to save NIE scores
NIE_paths, _ = geting_NIE_paths(config, method_name, treatments)
for path in counterfactual_paths: 
    component = path.split('/')[-1].split('_')[1 if config["is_averaged_embeddings"] else 2]
    print(f"{component}: {path}")

print(f'NIE_paths: {NIE_paths}')

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
# eval counterfactual
if args.eval_counterfactuals:
    evalutate_counterfactual(experiment_set, 
                             exp_loader, 
                             config, 
                             model, 
                             tokenizer, 
                             config['label_maps'], 
                             DEVICE, 
                             all_model_paths, 
                             summarize=True, 
                             DEBUG=True)

if args.collect_counterfactuals:
    # used getitem ExperimentDataset
    collect_counterfactuals(model, 
                            model_path, 
                            dataset_name, 
                            method_name, 
                            seed, 
                            counterfactual_paths, 
                            config, 
                            experiment_set, 
                            exp_loader, 
                            tokenizer, 
                            DEVICE=DEVICE) 

if args.compute_nie_scores:
    cma_analysis(config, 
                model_path,
                method_name, 
                seed, 
                counterfactual_paths, 
                NIE_paths, 
                save_nie_set_path = save_nie_set_path, 
                model = model, 
                treatments = treatments, 
                tokenizer = tokenizer, 
                experiment_set = experiment_set, 
                DEVICE = DEVICE, 
                DEBUG = True)   

if args.get_candidate_neurons: 
    get_candidate_neurons(config, method_name, NIE_paths, treatments=treatments, debug=False) 

if args.get_top_neurons_layer_each:
    from my_package.cma import get_top_neurons_layer_each
    get_top_neurons_layer_each(config, method_name, NIE_paths, treatments, debug=False)

if args.get_top_seq_candidate_neurons:
    from my_package.cma import get_sequential_neurons
    get_sequential_neurons(config, save_nie_set_path, counterfactual_paths, model_path, model, method_name, NIE_paths, tokenizer, DEVICE, debug=False)


