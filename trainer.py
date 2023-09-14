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
from utils import get_overlap_thresholds, group_by_treatment, test_mask, Classifier, get_hidden_representations
from utils import collect_output_components , report_gpu, trace_counterfactual
from utils import geting_counterfactual_paths, get_single_representation, geting_NIE_paths
from data import test_restore_weight
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
from data import ExperimentDataset, Dev, get_condition_inferences, get_inference_based, print_config, trace_optimized_params
from data import rank_losses, initial_partition_params, restore_original_weight, partition_param_train
from intervention import intervene, high_level_intervention
from analze import cma_analysis, compute_embedding_set, get_distribution, get_top_k
from utils import debias_test, get_nie_set_path
import yaml
from utils import get_num_neurons, get_params, get_diagnosis
from data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from data import exclude_grad
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
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available
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
from data import CustomDataset

logger = logging.get_logger(__name__)

class CustomTrainer(Trainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,):
        super().__init__(model, 
                         args, 
                         train_dataset= train_dataset, 
                         eval_dataset= eval_dataset, 
                         tokenizer =tokenizer, 
                         compute_metrics=compute_metrics,
                         optimizers = optimizers,
                         data_collator=data_collator,) 
        
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = CustomLabelSmoother(epsilon=self.args.label_smoothing_factor)
            print(f'label smoothing factor : {self.args.label_smoothing_factor}')
        else:
            print(f'label smoother is None :')
            self.label_smoother = None

        self._loss = torch.nn.CrossEntropyLoss()
    
    # Todo: custom where scheduler being created
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        cut_frac = 0.06

        # bugs: lr_scheuduler is no get_lr()[0]
        # Scheduler -> slanted override 
        # Allennlp -> override from
        if self.lr_scheduler is None:
            self.lr_scheduler = SlantedTriangular(optimizer = self.optimizer, 
                                           num_epochs = self.args.num_train_epochs,
                                           num_steps_per_epoch = len(self.get_train_dataloader()),
                                           cut_frac= cut_frac,
                                           )
            self._created_lr_scheduler = True

        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length: 
            import datasets
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None

            return BucketBatchSampler(batch_size= self.args.train_batch_size * self.args.gradient_accumulation_steps, 
                                      dataset=self.train_dataset,
                                      lengths=lengths,
                                      drop_last=False,
                                      model_input_name=model_input_name,
                                      DEBUG=True)
        else:
            return RandomSampler(self.train_dataset) # in __iter__() -> yeild the same as Batchsampler and bucket iterator
    
# def get_dataset(config, data_name = 'train_data'):
#     df = pd.read_json(os.path.join(config['data_path'], config[data_name]), lines=True)
#     df = preprocss(df)
#     df_new = df[['sentence1', 'sentence2', 'gold_label']]
#     df_new.rename(columns = {'gold_label':'label'}, inplace = True)
#     df_new['label'] = df_new['label'].apply(lambda label_text: to_label_id(label_text))
#     return Dataset.from_pandas(df_new)

def compute_metrics(eval_pred):
    # why compute_metrics never be called?
    # why there are problem in compute_metrics
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    global tokenizer
    global label_maps
    global metric


    with open("baseline_config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    dataset = {}
    tokenized_datasets = {}
    output_dir = '../models/baseline/' 
    label_maps = {"entailment": 0, "contradiction": 1, "neutral": 2}
    
    # random seed
    seed = config['seed'] #random.randint(0,10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    output_dir = os.path.join(output_dir, "seed_"+ str(seed))
    if not os.path.exists(output_dir): os.mkdir(output_dir) 
    metric = evaluate.load(config["validation_metric"])
    model_config = BertConfig(config['model']["model_name"])
    model_config.num_labels = len(label_maps.keys())
    tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
    model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = {}
    tokenized_datasets = {}
    
    for data_name in ["train_data", "validation_data", "test_data"]:
        print(f'========= {data_name} ===========')
        tokenized_datasets[data_name] = CustomDataset(config, label_maps=label_maps, data_name=data_name)
     
    training_args = TrainingArguments(output_dir = output_dir,
                                      report_to="none",
                                      overwrite_output_dir = True,
                                      evaluation_strategy=config['evaluation_strategy'],
                                      eval_steps=config['eval_steps'],
                                      learning_rate = float(config['optimizer']['lr']),
                                      weight_decay = config['optimizer']['weight_decay'],
                                      per_device_train_batch_size = config["data_loader"]["batch_sampler"]["batch_size"],
                                      per_device_eval_batch_size=config["data_loader"]["batch_sampler"]["batch_size"],
                                      num_train_epochs = config["num_epochs"],
                                      seed=seed,
                                      load_best_model_at_end=config["load_best_model_at_end"],
                                      save_total_limit= config["save_total_limit"],
                                      half_precision_backend = config["half_precision_backend"],
                                      group_by_length = config["data_loader"]["batch_sampler"]["group_by_length"],
                                     )
    
    opitmizer = AdamW(params=model.parameters(),
                      lr= float(config['optimizer']['lr']) , 
                      weight_decay = config['optimizer']['weight_decay'])

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
    # test_bucket_iterator(tokenized_datasets['train_data'])
    # Todo:
    # 1. fix bucket iterator
    # 2. change dataset to iterable-style and try yield 
    # 3. 
    

if __name__ == "__main__":
    main()
     
