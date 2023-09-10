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
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data.sampler import SequentialSampler            
from trainer_pt_utils import RandomSampler, SequentialSampler, BucketBatchSampler, BatchSampler, LengthGroupedSampler
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model


class CustomTrainer(Trainer):
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
            # Todo: recheck how huggingface sampler works?
            # sampler = SequentialSampler(self.train_dataset) 
            # BucketBatchSampler(sampler, batch_size=self.args.train_batch_size, drop_last=False)
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset) # in __iter__() -> yeild the same as Batchsampler and bucket iterator
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True) 

def preprocss(df):
    if '-' in df.gold_label.unique(): 
        df = df[df.gold_label != '-'].reset_index(drop=True)

    return df

def to_label_id(text_label): return label_maps[text_label]

def get_dataset(config, data_name = 'train_data'):
    df = pd.read_json(os.path.join(config['data_path'], config[data_name]), lines=True)
    df = preprocss(df)
    df_new = df[['sentence1', 'sentence2', 'gold_label']]
    df_new.rename(columns = {'gold_label':'label'}, inplace = True)
    df_new['label'] = df_new['label'].apply(lambda label_text: to_label_id(label_text))
    return Dataset.from_pandas(df_new)

def compute_metrics(eval_pred):
    # why compute_metrics never be called?
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
    for data_name in ["train_data", "validation_data", "test_data"]:
        print(f'========= {data_name} ===========')
        dataset[data_name] = get_dataset(config, data_name = data_name)
        tokenized_datasets[data_name] = dataset[data_name].map(tokenize_function, batched=True)
        if data_name == 'train_data': tokenized_datasets[data_name].shuffle(seed=seed)
     
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

if __name__ == "__main__":
    main()
     
