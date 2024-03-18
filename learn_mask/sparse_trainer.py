import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import IterableDataset, RandomSampler, Sampler
from transformers.utils import logging
from trainer_package.trainer_pt_utils import RandomSampler, SequentialSampler, BucketBatchSampler, BatchSampler, LengthGroupedSampler
import math
import time
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
import evaluate
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import allennlp
from trainer_package.slanted_triangular import SlantedTriangular
import numpy as np

from typing import Dict
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from collections import defaultdict
import torch.cuda
import torch.nn as nn
from learn_mask.utils import get_logger
import copy
from dataclasses import dataclass
# from transformers import AdamW, get_linear_schedule_with_warmup

log = get_logger(__name__)

class SparseTrainer(Trainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 sparse_args: SparseTrainingArguments = None,
                 freeze_weights: bool = False,
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
        
        # Todo: chage to reweighting loss
        self._loss = torch.nn.CrossEntropyLoss()

        #  ********* Sparse feature ***********
        self.sparse_args = sparse_args 
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = DEVICE
        self.model_patcher = ModelPatchingCoordinator(
                sparse_args=sparse_args,
                device=DEVICE,
                cache_dir="tmp/",  # Used only for teacher
                logit_names="logits",  # TODO
                teacher_constructor=None,  # TODO
        )

        self.model_patcher.patch_model(model)
        self.model = model
        self.freeze_weights = freeze_weights
        self.model = self.model.to(self.device)

        if self.freeze_weights:
            self.freeze_non_mask()
        
        # create optimizer  
        self.configure_optimizers()

    def freeze_non_mask(self):
        for name, param in self.model.named_parameters():
            if name.split(".")[-1] != "mask_scores":
                # freezing weight
                param.requires_grad = False
            else:
                # train mask only
                print(f'{name} : {param.shape}')

     
    def compile_model(self, model_path):
        """Returns compiled copy of a debiaed model (NOT in place)."""
        model = copy.deepcopy(self.model)
        model.load_state_dict(torch.load(model_path))
        removed, heads = self.model_patcher.compile_model(model)

        print(f"Compiled model. Removed {removed} / {heads} heads.")

        return model 

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

            from trainer_package.trainer_pt_utils import BucketIteratorAllennlp

            return BucketIteratorAllennlp(batch_size= self.args.train_batch_size * self.args.gradient_accumulation_steps, 
                                      dataset=self.train_dataset,
                                      lengths=lengths,
                                      max_len= self.tokenizer.model_max_length,
                                      seed=self.args.seed,
                                      drop_last=False,
                                      sorting_key=model_input_name,
                                      DEBUG=True)
            
        else:
            return RandomSampler(self.train_dataset) 
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        self.model_patcher.schedule_threshold(
            step=self.state.global_step,
            total_step=self.state.max_steps,
            training=True,
        )

        #{'threshold': 8.145975883902534e-06, 'regu_lambda': 0.0008145975883902534, 'ampere_temperature': 0.0016291951767790636}
        logs = self.model_patcher.log()
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            prune_reg_loss, _, _ = self.model_patcher.regularization_loss(model)
            # loss = -loss # like PCGU
            # self._loss(outputs['logits'], inputs['labels'].long().view(-1))
            
            # combine reg loss with total loss
            loss += prune_reg_loss

        # if self.state.global_step == 300:
        #     breakpoint()

        
        return (loss, outputs) if return_outputs else loss 

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:

            #0.2 * 12272 ~ 2454.4 -> 1 epoch
            total_steps = 147264
            num_warmup_steps = int(0.2 * total_steps)
            print(f'Total steps : {total_steps}, Num warmp steps : {num_warmup_steps}')
            # recheck warmup_steps
            self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
            )
            self._created_lr_scheduler = True
        
        return self.lr_scheduler
    
    def configure_optimizers(self):
        # bz: 32, learning rate of 3e-5, weight decay of 0.1,
        # and 20% warmup steps.
        training_args = MockTrainingArgs( learning_rate= self.args.learning_rate, weight_decay=self.args.weight_decay)
        optim_groups = self.model_patcher.create_optimizer_groups( self.model, args=training_args, sparse_args=self.sparse_args)

        optimizer = AdamW(optim_groups)
        self.optimizer = optimizer

@dataclass
class MockTrainingArgs:
    """Needed for calling model_patcher.create_optimizer_groups."""

    learning_rate: float
    weight_decay: float

# Todo:
# 1. [x] use focal loss follow clark in forward call
# 2. before call forward:
#    [x] call schedule_threshold with the current time stept 
#    [ ] call the log function to add pruning specific information
# 3. before calling backward:
#    [x] call regularization_loss, the first returned value is the regularization loss, other are used for logging

# Todo: learning rate didnt change
# 1. recheck scheduler linear warm up
# 2. recheck optmizer call (Adam)
# 3. recheck optimizer group from Patch_coordinator
