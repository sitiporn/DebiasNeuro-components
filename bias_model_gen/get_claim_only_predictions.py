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
from optimization import exclude_grad
from transformers import Trainer
# from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, AdamW, DataCollatorWithPadding
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
from data import FeverDatasetClaimOnly
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
from data import FeverDatasetClaimOnly
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

config_path = "./configs/baseline_config_fever.yaml"
with open(config_path, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

dataset = {}
tokenized_datasets = {}
output_dir = '../models/claimonly_fever/' 
label_maps = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

# random seed
seed = config['seed'] #random.randint(0,10000)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# "validation_metric": "+accuracy"?
metric = evaluate.load(config["validation_metric"])
model_config = BertConfig(config['model']["model_name"])
model_config.num_labels = len(label_maps.keys())
tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
model.to(DEVICE)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if config['data_loader']['batch_sampler']['dynamic_padding'] else None

tokenized_datasets = {}
for data_name in ["train_data", "validation_data", "test_data"]:
    print(f'========= {data_name} ===========')
    tokenized_datasets[data_name] = FeverDatasetClaimOnly(config, label_maps=label_maps, data_name=data_name)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(tokenized_datasets["train_data"], shuffle=True, batch_size=64, collate_fn=data_collator)
bias_probs = []
for batch in tqdm(train_dataloader):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        bias_probs.extend(F.softmax(outputs.logits.cpu(), dim=-1).tolist())

ori_dataset = {}
ori_dataset['train'] = pd.read_json('../data/fact_verification/fever.train.jsonl', lines=True)
ori_dataset['train']['bias_probs'] = bias_probs
temp_json = ori_dataset['train'].to_json(orient='records', lines=True)
with open('fever_claim_only.'+'train'+'.jsonl', 'w') as json_file:
    json_file.write(temp_json)