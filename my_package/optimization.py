import pickle
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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import sys
import operator
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from my_package.data import get_mediators, get_hidden_representations, get_specific_component, Dev, group_layer_params 
from transformers import AutoTokenizer, BertForSequenceClassification
from functools import partial
from my_package.optimization_utils import masking_grad, reverse_grad, initial_partition_params, trace_optimized_params


def intervene_grad(model, hooks, method_name, config, collect_param=False, DEBUG = 0):
    seed = config['seed']
    component_mappings = {}
    restore_path = f'../pickles/restore_weight/{method_name}/'
    value = config['k'] / 100 if config['k'] is not None else config['top_neuron_num']
    mode = config['top_neuron_mode']
    restore_path = os.path.join(restore_path, f'masking-{value}')
    mediators  = get_mediators(model)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v
    acc_train_num = 0
    acc_frozen_num = 0 
    grad_direction = config['grad_direction']

    #  walking in Encoder's parameters
    for param_name, param in model.named_parameters(): 
        splited_name = param_name.split('.')
        if 'encoder' not in splited_name: continue
        if 'LayerNorm' in splited_name: continue
        
        child = splited_name[-1]
        layer_id, component = get_specific_component(splited_name, component_mappings) 
        freeze_param_name = splited_name[-1]
        if mode == 'sorted':
            cur_restore_path = os.path.join(restore_path, f'{seed}_layer{layer_id}_collect_param={collect_param}_components.pickle')
        elif mode == 'random':
            cur_restore_path = os.path.join(restore_path, f'{seed}_radom_layer{layer_id}_collect_param={collect_param}_components.pickle')
        
        with open(cur_restore_path, 'rb') as handle:
            layer_params = pickle.load(handle)

        # group by components 
        train_neuron_ids = group_layer_params(layer_params, mode='train')
        frozen_neuron_ids = group_layer_params(layer_params, mode='freeze')
        
        # swap positions between train and freeze
        # train_neuron_ids = group_layer_params(layer_params, mode='freeze')
        # frozen_neuron_ids = group_layer_params(layer_params, mode='train')

        frozen_num =  len(frozen_neuron_ids[component]) if component in frozen_neuron_ids.keys() else 0
        train_num =  len(train_neuron_ids[component]) if component in train_neuron_ids.keys() else 0
        acc_train_num += train_num
        acc_frozen_num += frozen_num

        print(f'checking:{param_name}, frozen: {frozen_num}, train: {train_num}, Total : {frozen_num + train_num} : {param.shape}')
        assert frozen_num + train_num == param.shape[0]

        from my_package.optimization import reverse_grad
        if 'dense' in splited_name:
            if child == 'weight': 
                if component in list(train_neuron_ids.keys()): hooks.append(mediators[component](int(layer_id)).dense.weight.register_hook(partial(reverse_grad, grad_direction, train_neuron_ids[component], param_name, DEBUG)))
            elif child == 'bias':
                if component in list(train_neuron_ids.keys()):  hooks.append(mediators[component](int(layer_id)).dense.bias.register_hook(partial(reverse_grad, grad_direction, train_neuron_ids[component], param_name, DEBUG)))
            print(f'reverse grad dense : {param_name}') 
        else: 
            if child == 'weight': 
                if component in list(train_neuron_ids.keys()):  hooks.append(mediators[component](int(layer_id)).weight.register_hook(partial(reverse_grad, grad_direction, train_neuron_ids[component], param_name, DEBUG )))
            elif child == 'bias':
                if component in list(train_neuron_ids.keys()):  hooks.append(mediators[component](int(layer_id)).bias.register_hook(partial(reverse_grad, grad_direction, train_neuron_ids[component], param_name, DEBUG)))
            print(f'reverse grad  : {param_name}')
        

        # masking grad hooks : 144
        # reverse grad hooks : 134
    print(f'Reverse grad mode: {mode}')
    print(f"Gradient directoin: {grad_direction}")
    print(f'#Total train  neuron : {acc_train_num // 2}')
    print(f'#Total frozen neuron : {acc_frozen_num // 2}')
    
    return model, hooks

def restore_original_weight(model, DEBUG = False):
    
    value = 0.05
    count_freeze_params = 0
    component_mappings = {}
    restore_path = f'../pickles/restore_weight/'
    restore_path = os.path.join(restore_path, f'v-{value}')
    mediators  = get_mediators(model)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v

    #  walking in 
    for name, param in model.named_parameters(): 
        splited_name = name.split('.')
        if 'encoder' not in splited_name: continue
        if 'LayerNorm' in splited_name: continue
        # t.set_description(f"{name}")

        layer_id, component = get_specific_component(splited_name, component_mappings) 
        
        freeze_param_name = splited_name[-1]

        cur_restore_path = os.path.join(restore_path, f'layer{layer_id}_components.pickle')
        
        with open(cur_restore_path, 'rb') as handle:
            layer_params = pickle.load(handle)

        # Todo: vectorize accessing model parameters 
        for neuron_id in range(param.shape[0]):
            cur_comb = f'{component}-{neuron_id}' 
            # restore weight after performing optimize freeze param
            if cur_comb in list(layer_params.params[freeze_param_name].keys()):
                # modifying to restore original weight back 
                with torch.no_grad():
                    param[neuron_id] = layer_params.params[freeze_param_name][cur_comb]
                    count_freeze_params += 1
                    
    return model

def partition_param_train(model, tokenizer, config, do, counterfactual_paths, DEVICE, DEBUG=False):
    epochs = 3
    learning_rate = 2e-5
    grad_direction = None # should be matrix to perform elemense wise by sample 
    criterion = nn.CrossEntropyLoss(reduction = 'none') 
    optimizer = Adam(model.parameters(), lr= learning_rate)
    dev_set = Dev(config['dev_path'], config['dev_json'])
    dev_loader = DataLoader(dev_set, batch_size = 32, shuffle = False, num_workers=0)
    model = initial_partition_params(config, model, do) 
    hooks = []
    # when performing back propagation model it seems register o  ?
    model, hooks = intervene_grad(model, hooks=hooks)
    print(f'Epochs : {epochs}, with learning rate at : {learning_rate}')

    if DEBUG: 
        for name, param in model.named_parameters(): 
            if param.requires_grad == False: 
                print(f'freeze params state : {name}')
    
    
    if DEBUG: 
        print(f'Before optimize model {model.bert.pooler.dense.weight[:3, :3]}')
    
    # todo:
    # 2. collect loss for each step
    # 3. plot losses 
    losses = []
    accuracies = []
    
    for epoch in (e:= tqdm(range(epochs))):
        running_loss = 0.0
        
        for batch_idx, (inputs) in  enumerate(b:= tqdm(dev_loader)):

            model.train()
            cur_inputs = {} 

            for idx, (cur_inp, cur_col) in enumerate(zip(inputs, list(dev_set.df.keys()))): cur_inputs[cur_col] = cur_inp

            # get the inputs 
            pair_sentences = [[premise, hypo] for premise, hypo in zip(cur_inputs['sentence1'], cur_inputs['sentence2'])]
            pair_sentences = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
            pair_sentences = {k: v.to(DEVICE) for k,v in pair_sentences.items()}

            # ignore label_ids when running experiment on hans
            label_ids = torch.tensor([config['label_maps'][label] for label in cur_inputs['gold_label']]) if config['dev-name'] != 'hans' else None 
            
            # ignore label_ids when running experiment on hans
            if label_ids is not None: label_ids = label_ids.to(DEVICE)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # Todo: generalize to distribution if the storage is enough
            outs =  model(**pair_sentences, labels= label_ids if config['dev-name'] != 'hans' else None)

            loss = criterion(outs.logits, label_ids)
            test_loss = torch.mean(loss)
            
            scalers = cur_inputs['weight_score'] if config["dev-name"] == 'reweight' else torch.ones_like(loss)
            scalers = scalers.to(DEVICE)

            assert abs(outs.loss - test_loss) < 1e-6
            assert scalers.shape == loss.shape
            
            # loss =  torch.mean(scalers * loss * grad_direction) 
            loss =  torch.mean(scalers * loss) 
            loss.backward()

            optimizer.step()                
            
            trace_optimized_params(model, config, DEVICE, DEBUG=True)

            if DEBUG: print(f'{model.bert.pooler.dense.weight[:3, :3]}')

            # print statistics
            running_loss += loss.item()
            losses.append(loss.item())
            
            if batch_idx % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        SAVE_MODEL_PATH = f'../pickles/models/reweight_model_partition_params_epoch{epoch}.pth'
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f'save model into {SAVE_MODEL_PATH}')
    
    if DEBUG: 
        print(f'After optimize model {model.bert.pooler.dense.weight[:3, :3]}')
        print(f'pooler requires grad {model.bert.pooler.dense.weight.requires_grad}')

    with open(f'../pickles/losses/{config["dev-name"]}.pickle', 'wb') as handle: 
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saving losses into pickle files')

import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer

class CustomAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 1e-3):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        config,
        params: Iterable[nn.parameter.Parameter],
        original_model: Iterable[nn.parameter.Parameter],
        seed,
        collect_param,
        method_name,
        DEVICE,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        from transformers.utils.versions import require_version
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.original_model = original_model
        value = config['k'] / 100 if config['k'] is not None else config['top_neuron_num'] 
        restore_path = f'../pickles/restore_weight/{method_name}/'
        self.restore_path = os.path.join(restore_path, f'masking-{value}')
        self.seed = seed 
        self.collect_param = collect_param
        self.component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
        mediators  = get_mediators(self.original_model)
        component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
        self.component_mappings = {}
        for k, v in zip(component_keys, mediators.keys()): self.component_mappings[k] = v
        self.DEVICE = DEVICE
        self.config = config
        print(f'Custom optimizer freze mode :{self.config["top_neuron_mode"]}')

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        acc_train_num  = 0 
        acc_frozen_num = 0
        
        for group in self.param_groups:
            for p, (param_name, param) in zip(group["params"], self.original_model.named_parameters()):
                splited_name = param_name.split('.')
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                
                child = splited_name[-1]
                layer_id, component = get_specific_component(splited_name, self.component_mappings) 
                freeze_param_name = splited_name[-1]
                
                if self.config["top_neuron_mode"] == 'sorted':
                    cur_restore_path = os.path.join(self.restore_path, f'{self.seed}_layer{layer_id}_collect_param={self.collect_param}_components.pickle')
                elif self.config["top_neuron_mode"] == 'random':
                    cur_restore_path = os.path.join(self.restore_path, f'{self.seed}_radom_layer{layer_id}_collect_param={self.collect_param}_components.pickle')

                with open(cur_restore_path, 'rb') as handle: layer_params = pickle.load(handle)
                frozen_neuron_ids = group_layer_params(layer_params, mode='freeze')
                train_neuron_ids  = group_layer_params(layer_params, mode='train')
                
                neuron_ids = []
                # neuron_ids = [*range(0, param.shape[0], 1)]
                neuron_ids += frozen_neuron_ids[component] if component in frozen_neuron_ids.keys() else []
                # neuron_ids += train_neuron_ids[component] if component in train_neuron_ids.keys() else []
                mask = torch.ones_like(p)
                mask[neuron_ids] = 0

                # if true p and false set it original param
                # p: current updated parameters
                # weight : [neuron_num, input_size], bias : [num_neurons]
                p.data = torch.where(mask.bool(), p, param.to(self.DEVICE))
                assert (p[neuron_ids] == param[neuron_ids].to(self.DEVICE)).all(), f'param_name : {param_name}'
                
                # extra debugs
                # frozen_num =  len(frozen_neuron_ids[component]) if component in frozen_neuron_ids.keys() else 0
                # train_num =  len(train_neuron_ids[component]) if component in train_neuron_ids.keys() else 0
                # acc_train_num += train_num
                # acc_frozen_num += frozen_num
                # print(f'customA:{param_name}, frozen: {frozen_num}, train: {train_num}, Total : {frozen_num + train_num} : {param.shape}')

        return loss

def test_grad_zero():
    from transformers import TrainingArguments, Trainer
    import yaml
    import copy

    config_path = "./configs/pcgu_config.yaml"
    with open(config_path, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    from my_package.optimization import CustomAdamW
    lr = 1e-2 
    # # Setup
    model = nn.Conv2d(1, 10, 3, 1, 1)
    original_model = copy.deepcopy(model)
    # weight_reference = model.weight.clone()
    
    optimizer = CustomAdamW(params=model.parameters(),
                            original_model= original_model,
                            # seed = seed,
                            # method_name=method_name,
                            # DEVICE=DEVICE,
                            # collect_param=config['collect_param'],
                            lr= lr , 
                            weight_decay = config['optimizer']['weight_decay'])

    # optimizer = torch.optim.SGD(model.parameters(), lr=1., momentum=0.0)

    # Should fail
    # model(torch.randn(1, 1, 3, 3)).mean().backward()
    # optimizer.step()
    # print(f'without zero grads : {(model.weight == weight_reference).all()}')

    # Should work with momentum = 0.0
    optimizer.zero_grad()
    weight_reference = model.weight.clone()
    model.weight.register_hook(lambda grad: grad * 0)
    # model.weight.register_post_accumulate_grad_hook(lambda grad: grad * 0)
    model(torch.randn(1, 1, 3, 3)).mean().backward()
    print(f'with zero grads : {(model.weight == weight_reference).all()}')
    optimizer.step()
    
    # fail because pytorch version there is register_post_accumulate_grad_hook
    # v = torch.tensor([0., 0., 0.], requires_grad=True)
    # lr = 0.01
    # # simulate a simple SGD update
    # h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))
    # v.backward(torch.tensor([1., 2., 3.]))
    # print(v)
    # h.remove()  # removes the hook
    # breakpoint()

# test_grad_zero()