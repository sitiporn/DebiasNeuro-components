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
from my_package.optimization_utils import trace_optimized_params, initial_partition_params
from my_package.optimization import partition_param_train, restore_original_weight

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
from my_package.data import ExperimentDataset, Dev, print_config
from my_package.data import rank_losses

from my_package.intervention import intervene, high_level_intervention
# from my_package.cma import cma_analysis, get_distribution, get_top_k
from my_package.utils import debias_test
from my_package.cma_utils import get_nie_set_path
import yaml
from my_package.utils import get_num_neurons, get_params, get_diagnosis
from my_package.data import get_analysis 
from transformers import AutoTokenizer, BertForSequenceClassification
from my_package.optimization import exclude_grad
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

from custom_model import BertForSequenceClassificationReweight
logger = logging.get_logger(__name__)

def get_max_padding_lenght(input_ids:torch.Tensor):
    # get the lowest index in each row
    id_poses = []
    for i in range(input_ids.shape[0]):
        ids = ((input_ids[i,:] == 0).nonzero(as_tuple=True)[0])
        if ids.shape[0] != 0: id_poses.extend(ids.tolist())
    #find the lowest one
    if len(id_poses) > 0:
        start_pos = sorted(id_poses)[0] 
        max_pad_len = abs(start_pos - input_ids.shape[-1])
    else:
        max_pad_len = 0
    print(f'padding len : {max_pad_len}')

class ReweightTrainer(Trainer):
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

            from trainer_package.trainer_pt_utils import BucketIteratorAllennlp

            return BucketIteratorAllennlp(batch_size= self.args.train_batch_size * self.args.gradient_accumulation_steps, 
                                      dataset=self.train_dataset,
                                      lengths=lengths,
                                      max_len= self.tokenizer.model_max_length,
                                      seed=self.args.seed,
                                      drop_last=False,
                                      sorting_key=model_input_name,
                                      DEBUG=True)
            # LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
            #                            dataset=self.train_dataset,
            #                            lengths=lengths,
                                    #    model_input_name=model_input_name,)
             ## BucketBatchSampler(batch_size= self.args.train_batch_size * self.args.gradient_accumulation_steps, 
            #                           dataset=self.train_dataset,
            #                           lengths=lengths,
            #                           drop_last=False,
            #                           model_input_name=model_input_name,
            #                           DEBUG=True)
            
        else:
            return RandomSampler(self.train_dataset) 
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # breakpoint()
        from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # get_max_padding_lenght(input_ids=inputs['input_ids'])
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # from my_package.intervention import compute_nie
        outputs = model(**inputs)
        # Todo: get nie scores
        # Todo: get nie scores
        # compute_nie(inputs, counterfactual=None, model=model)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # breakpoint()
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # self._loss(outputs['logits'], inputs['labels'].long().view(-1))

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    global tokenizer
    global label_maps
    global metric


    config_path = "./configs/reweight_config.yaml"
    with open(config_path, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    dataset = {}
    tokenized_datasets = {}
    output_dir = '../models/reweight/' 
    label_maps = {"entailment": 0, "contradiction": 1, "neutral": 2}
    
    # random seed
    seed = config['seed'] #random.randint(0,10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    output_dir = os.path.join(output_dir, "seed_"+ str(seed))
    if not os.path.exists(output_dir): os.mkdir(output_dir) 
    # "validation_metric": "+accuracy"?
    metric = evaluate.load(config["validation_metric"])
    model_config = BertConfig(config['model']["model_name"])
    model_config.num_labels = len(label_maps.keys())
    tokenizer = AutoTokenizer.from_pretrained(config['tokens']['model_name'], model_max_length=config['tokens']['max_length'])
    model = BertForSequenceClassificationReweight.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if config['data_loader']['batch_sampler']['dynamic_padding'] else None
    dataset = {}
    tokenized_datasets = {}

    for data_name in ["train_data", "validation_data", "test_data"]:
        print(f'========= {data_name} ===========')
        tokenized_datasets[data_name] = CustomDataset(config, label_maps=label_maps, data_name=data_name)
    print(f'Config of dataloader') 
    print(f'Group by len : {config["data_loader"]["batch_sampler"]["group_by_length"]}')
    print(f"Dynamics padding : {config['data_loader']['batch_sampler']['dynamic_padding']}, {data_collator}")
    
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
    
    opitmizer = AdamW(params=model.parameters(),
                    lr= float(config['optimizer']['lr']) , 
                    weight_decay = config['optimizer']['weight_decay'])

    # NOTE: for baseline just change 
    # config: baseline_config.yaml
    # model: BertForSequenceClassification

    trainer = ReweightTrainer(
        model,
        training_args,
        train_dataset= tokenized_datasets["train_data"],
        eval_dataset= tokenized_datasets["validation_data"],
        tokenizer =tokenizer, 
        # "validation_metric": "+accuracy"?
        compute_metrics=compute_metrics,
        optimizers = (opitmizer, None),
        data_collator=data_collator,
        )
    
   
    trainer.train()
    

if __name__ == "__main__":
    main()
     

