from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gc
import os
import os.path
from my_package.intervention import neuron_intervention
# from tabulate import tabulate
import statistics 
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from my_package.intervention import get_mediators
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertForSequenceClassification
from functools import partial
from sklearn.metrics import f1_score

def report_gpu(): 
#   print(f"++++++++++++++++++++++++++++++")
#   print(f"before deleting : {torch.cuda.list_gpu_processes()}")
  gc.collect()
  torch.cuda.empty_cache()
  print(f'clear cached!')
#   print(f"after emptying cache : {torch.cuda.list_gpu_processes()}")
#   print(f"++++++++++++++++++++++++++++++")
  

class BertAttentionOverride(nn.Module):
    """A copy of `modeling_bert.BertSelfAttention` class, but with overridden attention values"""

    def __init__(self, module, attn_override, attn_override_mask):
        """
        Args:
            module: instance of modeling_bert.BertSelfAttentionOverride
                from which variables will be copied
            attn_override: values to override the computed attention weights.
                Shape is [bsz, num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention weights to override.
                Shape is [bsz, num_heads, seq_len, seq_len]
        """
        super().__init__()
        # if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads)
        #     )
        self.output_attentions = True #module.output_attentions
        self.num_attention_heads = module.num_attention_heads
        self.attention_head_size = module.attention_head_size
        self.all_head_size = module.all_head_size
        self.query = module.query
        self.key = module.key
        self.value = module.value
        self.dropout = module.dropout
        # Set attention override values
        self.attn_override = attn_override
        self.attn_override_mask = attn_override_mask

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Intervention:
        # attn_override and attn_override_mask are of shape (batch_size, num_heads, override_seq_len, override_seq_len)
        # where override_seq_len is the length of subsequence for which attention is being overridden
        override_seq_len = self.attn_override_mask.shape[-1]

        # print("=====  Before : ")
        # print(attention_probs[:, :, :override_seq_len, :override_seq_len])
        attention_probs[:, :, :override_seq_len, :override_seq_len] = torch.where(
            self.attn_override_mask,
            self.attn_override,
            attention_probs[:, :, :override_seq_len, :override_seq_len])
        
        # print("=====  After : ")
        # print(attention_probs[:, :, :override_seq_len, :override_seq_len])
        
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        return outputs



def debias_test(do, 
                layer, 
                model, 
                experiment_set, 
                tokenizer,
                DEVICE, 
                layers, 
                heads,
                counterfactual_paths,
                label_maps,
                is_group_by_class, 
                is_averaged_embeddings, 
                intervention_type,
                debug = False):

    path = f'pickles/top_neurons/top_neuron_{do}_{layer}.pickle'

    nie_dataloader = None
    hooks = None
    
    interventions = ["Null", "Intervene"]
    distributions = {} 
    NIE = {}
    mediators = {}
    counters = {}

    for do in ['High-overlap','Low-overlap']: 

        distributions[do] = {}
        NIE[do] = {}
        counters[do] = 0
        
        for mode in interventions: 

            distributions[do][mode] = {}
            NIE[do][mode] = {}

            for golden in ['contradiction','entailment','neutral']:

                distributions[do][mode][golden] = []
                NIE[do][mode][golden] = []

    counterfactual_loader = DataLoader(experiment_set, 
                                       batch_size = 32,
                                       shuffle = False, 
                                       num_workers=0)
  
    with open(path, 'rb') as handle:
        # get [CLS] activation 
        top_neuron = pickle.load(handle)

    # ++++++++++++  for single neuron intervention ++++++++++++
    percent, nie = list(top_neuron.keys())[0], list(top_neuron.values())[0] 
    
    component = list(top_neuron[percent].keys())[0].split('-')[0]
    neuron_id  = int(list(top_neuron[percent].keys())[0].split('-')[1])
    
    print(f"component : {component}")
    print(f"neuron_id : {neuron_id}")
    
    # mediator used to intervene
    mediators  = get_mediators(model)

    cls = get_hidden_representations(counterfactual_paths, layers, heads, is_group_by_class, is_averaged_embeddings)
    
    Z = cls[component][do][layer]

    for batch_idx, (sentences, labels) in enumerate(counterfactual_loader):
            
        cur_dist = {}

        # for idx, do in enumerate(tqdm(['High-overlap','Low-overlap'], desc="Do-overlap")):
        for idx, do in enumerate(['High-overlap','Low-overlap']):

            premise, hypo = sentences[do]

            pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
            
            inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")

            inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

            cur_dist[do] = {}

            for mode in interventions:

                if mode == "Intervene": 

                    hooks = []
                    
                    hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(
                                                                                    neuron_ids = [neuron_id], 
                                                                                    component= component,
                                                                                    DEVICE = DEVICE ,
                                                                                    value = Z,
                                                                                    intervention_type=intervention_type)))

                with torch.no_grad(): 
                    
                    # Todo: generalize to distribution if the storage is enough
                    cur_dist[do][mode] = F.softmax(model(**inputs).logits , dim=-1)
                
                
                if mode == "Intervene": 
                    for hook in hooks: hook.remove() 

                for sample_idx in range(cur_dist[do][mode].shape[0]):

                    distributions[do][mode][labels[do][sample_idx]].append(cur_dist[do][mode][sample_idx,:])

            
            
            ret = torch.sum(cur_dist[do]["Intervene"][:,label_maps["entailment"]] / cur_dist[do]["Null"][:,label_maps["entailment"]],dim=0)
            ret = ret / inputs['input_ids'].shape[0]
            
            counters[do] += inputs['input_ids'].shape[0]

    for do in ['High-overlap','Low-overlap']:


        print(f"===================  {do}  ======================")
            
        for golden in ['contradiction','entailment','neutral']:

            print(f"++++++++++++++++  {golden}  ++++++++++++++++++++")
        
            for mode in interventions:

                distributions[do][mode][golden] = torch.stack(distributions[do][mode][golden], dim=0)

                predictions = torch.argmax(distributions[do][mode][golden],dim=-1).tolist()

                print(f"{mode} : {Counter(predictions)}")

def get_ans(ans: int):

    # Todo: generalize to all challenge  sets 
    if ans == 0:
        return "entailment"
    else:
        return "non-entailment"


def get_outliers(class_name, outliers,label_maps, data):
    # Todo: get outliers
    # Contradiction set; entailment > 100, neutral > 100 
    # entailment set; contradiction > 80, neutral > 100 
    # 
    data = data[:,label_maps[class_name]]

    Q1 = np.percentile(data, 25, interpolation = 'midpoint') 
    Q2 = np.percentile(data, 50, interpolation = 'midpoint') 
    Q3 = np.percentile(data, 75, interpolation = 'midpoint') 

    IQR = Q3 - Q1 

    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR

    print('Interquartile range is', IQR)
    print('Upper is', up_lim)
    print('Lower is', low_lim)
    
    # outlier =[]
    # outlier_idxes = []

    data = data.tolist()
    
    for idx, x in enumerate(data):
        
        if ((x> up_lim) or (x<low_lim)):
            # outlier.append(x)
            outliers.append(idx)

def print_distributions(cur_dist, label_maps, interventions, sample_idx):


    #results = [['Method', 'MNLI-dev-mm','MNLI-HANS','QQP-dev','QQP-PAWS']]
    headers = ['Mode']
    headers.extend(list(label_maps.keys()))
    table = []

    for mode in interventions:

            entry = [mode]
            
            for label in headers[1:]:

                entry.append(float(cur_dist[mode][sample_idx,label_maps[label]].cpu()))
            
            table.append(entry)
    
    print(tabulate(table[1:], headers, tablefmt="grid"))   


def new_features(v):
    print(v)
    return v
    
def load_model(path, model, device='cuda:0'):
    print(f'Loading model from {path}')
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model

def compute_acc(raw_distribution_path, label_maps):
    label_remaps = {v:k for k, v in label_maps.items()}
    with open(raw_distribution_path, 'rb') as handle: 
        distributions = pickle.load(handle)
        golden_answers = pickle.load(handle)
        print(f'loading distributions and labels from : {raw_distribution_path}')
    acc = {} 
    if isinstance(distributions, dict):
        modes = list(distributions.keys())
    else:
        modes = ['Null']
        distributions  = {'Null': distributions}
        golden_answers = {'Null': golden_answers}
    
    print(f'compute_acc modes:{modes}') # Intervene, Null
     
    for mode in modes:
        acc[mode] = {k: [] for k in (['all'] + list(label_maps.keys()))}
        for dist, label in zip(distributions[mode], golden_answers[mode]):
            prediction = int(torch.argmax(dist))
            acc[mode]['all'].append(prediction == int(label))
            # acc[label_remaps[label]].append(label_remaps[prediction] == label) 
            acc[mode][label_remaps[int(label)]].append(prediction == int(label))
        to_pop = []
        for key in acc[mode].keys():
            if len(acc[mode][key])==0:
                to_pop.append(key)
        for p in to_pop:
            acc[mode].pop(p, None)
        acc[mode] = { k: sum(acc[mode][k]) / len(acc[mode][k]) for k in list(acc[mode].keys())}
    
    return acc

def get_num_neurons(config):

    label_maps = config['label_maps']
    model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
    
    config['nums']['self'] = model.bert.encoder.layer[0].attention.self.query.out_features * config['attention_components']
    config['nums']['AO'] = model.bert.encoder.layer[0].attention.output.dense.out_features
    config['nums']['I'] = model.bert.encoder.layer[0].intermediate.dense.out_features
    config['nums']['O'] = model.bert.encoder.layer[0].output.dense.out_features
    num_layer = len(model.bert.encoder.layer)
    total_neurons = num_layer * (config['nums']['self'] + config['nums']['AO'] + config['nums']['I'] + config['nums']['O'])

    return total_neurons

def get_params(config, soft_masking_value_search:bool=False, masking_rate_search:bool=False):
    """hyperparameters to modify neuron's activations tuned on validation sets
       Args:
          config: config for experiment
          soft_masking_value_search: do hyperparameter search on value used modify neurons's activations
          masking_rate_search: do hyperparameter search on the percentage of neurons to mask
    """
    params = {}
    digits = {}
    search_hyper_types = []
    if soft_masking_value_search: search_hyper_types.append('epsilons')
    if masking_rate_search: search_hyper_types.append('percent')
    """
    if intervention_type == "weaken": output[:,CLS_TOKEN, neuron_ids] = output[:,CLS_TOKEN, neuron_ids] * epsilon
    elif intervention_type == "neg": output[:,CLS_TOKEN, neuron_ids] = output[:,CLS_TOKEN, neuron_ids] * -1
    elif intervention_type ==  'remove':
    """
    # masking rate : known, seach not epsilons
    # epsilons 1. weaken 2. remove
    for op in ['epsilons','percent']:
        low  =  config[op]['low'] 
        high =  config[op]['high']  
        step =  config[op]['step'] 
        digits[op] = len(str(step).split('.')[-1])
        size= config[op]['size']
        mode = config[op]['mode']
        # hyperparam search
        if op in search_hyper_types: 
            if op == 'epsilons':
                if config['intervention_type'] == "remove": params[op] = (low - high) * torch.rand(size) + high  # the interval (low, high)
                elif config['intervention_type'] == "weaken": params[op] = [round(val, digits[op]) for val in np.arange(low, high, step).tolist()]
            else:
                pass# search hyperparam on masking rate
        # know hyperparameters
        else:
            if op == 'percent': params[op] = [config['masking_rate']] 
            elif op == 'epsilons' and config['weaken_rate'] is not None: params[op] = [config['weaken_rate']]
            elif config['intervention_type'] not in ["remove","weaken"]: params[op] = [0]
    return  params

def get_diagnosis(config):
    """ This function is used to find upper bound of our methods"""

    print(f"{config['label_maps']}")
    
    SAMPLE_SIZE = 5
    
    for dev in ['mismatched', 'hans']:
        
        # get raw  distributions of intervention
        value = config["masking_rate"]

        key = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'
        layer = config['layer']
        do = 'High-overlap'

        params, digits = get_params(config)
        
        if layer == -1:
            raw_distribution_path = f'raw_distribution_{key}_{do}_all_layers_{value}-k_{config["intervention_type"]}_{dev}.pickle'  
        else:
            raw_distribution_path = f'raw_distribution_{key}_{do}_L{layer}_{value}-k_{config["intervention_type"]}_{dev}.pickle'

        prediction_path = 'pickles/prediction/' 
        epsilon_path = f"v{round(params['epsilons'][0], digits['epsilons'])}"
        
        raw_distribution_path = os.path.join(os.path.join(prediction_path, epsilon_path),  raw_distribution_path)

        # if dev == "hans": raw_distribution_path = 'pickles/prediction/v0.7/raw_distribution_0.7_High-overlap_all_layers_0.05-k_weaken_hans.pickle'
        with open(raw_distribution_path, 'rb') as handle: 
            
            distributions = pickle.load(handle)
            golden_answers = pickle.load(handle)
            losses = pickle.load(handle)

        # Todo: get index of current labels
        print(f" ++++++++  {dev} set, masking rate {value}, weaken rate : {key} ++++++++")
        # print(f"cur path : {raw_distribution_path}")
        
        for mode in ['Null', 'Intervene']:

            print(f">> {mode}")
        
            cur_labels = set(golden_answers[mode])
            
            dists = {}
        
            for idx, label in enumerate(golden_answers[mode]): 
                
                if label not in dists.keys(): dists[label] = []

                dists[label].append(distributions[mode][idx])
        
            for label in cur_labels:
                print(f"== {label} == ")
                print(torch.stack(dists[label])[:5].cpu())
            
    # raw_distribution_path : 

# 3 class {entailment: 0, contrandiction: 1, neatral: 2}
def relabel(label):

    if label == 'contradiction':
        return 1
    elif label == 'neutral':
        return 2
    else:
        return 0

# ps
def give_weight(label, probs): 

    golden_id = relabel(label)

    probs = probs[golden_id]

    return 1 / probs

class LayerParams:
    """ A class used to mangage parameters of each layer  to use in optimization step """
    def __init__(self, layer_id, num_train_params, num_freeze_params):
        self.layer_id  = layer_id
        self.train_params = {'weight': {}, 'bias': {}}
        self.freeze_params = {'weight': {}, 'bias': {}}
        self.num_train_params = num_train_params
        self.num_freeze_params = num_freeze_params
        self.total_params = self.num_train_params + self.num_freeze_params

    def append_frozen(self, pos, value):
        component = pos.split('-')[2]
        neuron_id = pos.split('-')[3]
        for child in list(self.freeze_params.keys()): 
            self.freeze_params[child][f'{component}-{neuron_id}'] = value[child].cpu() if value[child] is not None else None

    def append_train(self, pos, value):
        component = pos.split('-')[2]
        neuron_id = pos.split('-')[3]
        for child in list(self.train_params.keys()): 
            self.train_params[child][f'{component}-{neuron_id}'] = value[child].cpu() if value[child] is not None else None

def test_layer_params(encoder_params, freeze_params, train_params, value):

    train_param_count = {}
    freeze_param_count = {}

    for layer_id in range(len(encoder_params)):
        for child in ['weight', 'bias']:
            if child not in train_param_count.keys(): train_param_count[child] = 0
            if child not in freeze_param_count.keys(): freeze_param_count[child] = 0
            
            train_param_count[child] += len(encoder_params[layer_id].train_params[child].keys())
            freeze_param_count[child] +=  len(encoder_params[layer_id].freeze_params[child].keys())

    print(f'summary after combine parameters')
    for child in ['weight', 'bias']:
        assert train_param_count[child]   == encoder_params[layer_id].num_train_params  
        assert freeze_param_count[child]  == encoder_params[layer_id].num_freeze_params  
        assert encoder_params[layer_id].total_params == train_param_count[child] +  freeze_param_count[child]
        print(f'********** {child}  ************')
        print(f'# Train : {train_param_count[child]}')
        print(f'# Freeze : {freeze_param_count[child]}')
        print(f'# Total  : {train_param_count[child] +  freeze_param_count[child]}')

def compare_weight(updated_model, reference_model):
    for (_, updated_param)  , (param_name, ref_param) in zip(updated_model.named_parameters(), reference_model.named_parameters()):
        if not updated_param.requires_grad: continue
        assert (updated_param == ref_param).all(), f'{param_name} not exact match'
        print(f'{param_name}: {ref_param.shape}')

    print(f'Comparing weight checking Done!')

    
def compare_frozen_weight(LOAD_REFERENCE_MODEL_PATH, LOAD_MODEL_PATH, config, method_name, restore_path):
    from my_package.data import get_specific_component, group_layer_params, get_all_model_paths

    label_maps = config['label_maps'] 
    collect_param = config['collect_param']
    model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
    reference_model = BertForSequenceClassification.from_pretrained(config['tokens']['model_name'], num_labels = len(label_maps.keys()))
    value = config['k'] / 100
    restore_path = f'../pickles/restore_weight/{method_name}/'
    restore_path = os.path.join(restore_path, f'masking-{value}') 

    from my_package.utils import load_model
    all_model_paths = get_all_model_paths(LOAD_REFERENCE_MODEL_PATH)
    all_optimized_model_paths = get_all_model_paths(LOAD_MODEL_PATH)
    
    for seed, _ in all_model_paths.items():
        path = all_model_paths[str(seed)]
        optimized_path = all_optimized_model_paths[str(seed)]
        optimized_model = load_model(path=path, model=model)
        reference_model = load_model(path=optimized_path, model=reference_model)
        print(f'Loading updated model from : {optimized_path} to optimize on PCGU done!')
        print(f'Loading reference model from : {path} done!')
        
        component_mappings = {}
        mediators  = get_mediators(optimized_model)
        component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
        for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v


        for (_, optimized_param)  , (param_name, ref_param) in zip(optimized_model.named_parameters(), reference_model.named_parameters()):
            splited_name = param_name.split('.') 
            child = splited_name[-1]
            
            if 'encoder' not in splited_name or 'LayerNorm' in splited_name: 
                assert (optimized_param == ref_param).all() , f'{param_name}'
                continue
            
            layer_id, component = get_specific_component(splited_name,component_mappings) 
            freeze_param_name = splited_name[-1]
            cur_restore_path = os.path.join(restore_path, f'{seed}_layer{layer_id}_collect_param={collect_param}_components.pickle')
            with open(cur_restore_path, 'rb') as handle: layer_params = pickle.load(handle)
            frozen_neuron_ids = group_layer_params(layer_params, mode='freeze')
            train_neuron_ids = group_layer_params(layer_params, mode='train')
            
            neuron_ids = []
            neuron_ids += frozen_neuron_ids[component] if component in frozen_neuron_ids.keys() else []
            assert (optimized_param[neuron_ids] == ref_param[neuron_ids]).all() , f'{param_name}: {optimized_param[neuron_ids].shape}'


# def prunning_neurons(neuron_ids:int, m:torch.Tensor, multiplier:float, param_name:str, DEBUG:bool):
#     def prunning_hook(module, input, output):
#         """ Hook for prunning output during forward pass """
#         # out_dim:  ([bz, seq_len, neuron_num])
#         mask = torch.ones_like(output)
#         # prunning input tensor respect to candidate neurons
#         mask[:,:, neuron_ids] = 0 if m.shape[0] == 0 else m
#         mask = multiplier * mask
#         if DEBUG and len(neuron_ids) > 0:
#             print(f'{param_name}, #bias neurons: {len(neuron_ids)}')
#             print(f'inp mean: {output.mean()}, out mean:{(output * mask).mean()} mask mean: {mask.mean()}, multiplier: {multiplier}')
#             test_ids = [1,2,3,5]
#             x = torch.rand(50000, 10)
#             test_mask = torch.ones_like(x)
#             test_ratio = x.shape[-1] / (x.shape[-1] - len(test_ids) )
#             test_mask[:,test_ids] = 0
#             test_mask = test_mask * test_ratio
#             out = x * test_mask
#             print(f'test result, inp mean: ({x.mean():.4f}), out mean: ({out.mean():.4f}).')

#         return output * mask
#     return prunning_hook

def prunning_neurons(neuron_ids:int, m:torch.Tensor, multiplier:float, param_name:str, DEBUG:bool):
    def prunning_hook(module, input, output):
        """ Hook for prunning output during forward pass """
        # out_dim:  ([bz, seq_len, neuron_num])
        CLS_TOKEN = 0
        # bz, seq_len, hidden_dim
        output[:,CLS_TOKEN, neuron_ids] = output[:,CLS_TOKEN, neuron_ids] * 0.9
    return prunning_hook


def prunning_biased_neurons(model, seed, NIE_paths, config, method_name, hooks, DEBUG=False, DEVICE="cuda:0"):
    from my_package.data import get_specific_component, group_layer_params, get_all_model_paths
    value = config['masking_rate']
    label_maps = config['label_maps'] 
    collect_param = config['collect_param']
    component_mappings = {}
    mediators  = get_mediators(model)
    component_keys = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    restore_path = os.path.join(restore_path, f'masking-{value}')
    for k, v in zip(component_keys, mediators.keys()): component_mappings[k] = v
    from my_package.cma import scaling_nie_scores
    nie_table_df = scaling_nie_scores(config, method_name, NIE_paths, debug=False)
    m = {row['Neuron_ids']: row['M_MinMax'] for index, row in nie_table_df.iterrows()} 
    
    for param_name, param in model.named_parameters():
        splited_name = param_name.split('.') 
        if 'encoder' not in splited_name or 'LayerNorm' in splited_name: continue
        child = splited_name[-1]
        layer_id, component = get_specific_component(splited_name,component_mappings) 
        freeze_param_name = splited_name[-1]
        cur_restore_path = os.path.join(restore_path, f'{seed}_layer{layer_id}_collect_param={collect_param}_components.pickle')
        with open(cur_restore_path, 'rb') as handle: layer_params = pickle.load(handle)
        frozen_neuron_ids = group_layer_params(layer_params, mode='freeze')
        biased_neuron_ids = group_layer_params(layer_params, mode='train')

        frozen_num =  len(frozen_neuron_ids[component]) if component in frozen_neuron_ids.keys() else 0
        train_num =  len(biased_neuron_ids[component]) if component in biased_neuron_ids.keys() else 0
        neuron_ids = biased_neuron_ids[component] if component in biased_neuron_ids.keys() else []

        num_neurons = config['num_neurons'] if config['criterion'] == 'num_neurons' else config["masking_rate"] * nie_table_df.shape[0]
        best_neuron_ids = nie_table_df['Neuron_ids'].iloc[:int(num_neurons)].tolist()
        
        neuron_ids = [neuron_id for neuron_id in neuron_ids if f'L-{layer_id}-{component}-{neuron_id}' in best_neuron_ids]
        
        nie_m = []
        for neuron_id in neuron_ids: 
            if f'{component}-{neuron_id}'  in layer_params.train_params[child].keys():
                nie_m.append(m[f'L-{layer_id}-{component}-{neuron_id}']) 

        if child == 'weight':
            sum_m = sum(nie_m) if len(nie_m) > 0 else 0 
            nie_m = len(nie_m) * [0.9]
            multiplier = param.shape[0] / (param.shape[0] - len(nie_m) + sum(nie_m) )
            # multiplier = param.shape[0] / (frozen_num + sum_m)
            nie_m = torch.tensor(nie_m).to(DEVICE)
            print(f'{param_name}, frozen:{frozen_num}, train:{train_num}, multiplier: {multiplier}, neuron_ids: {len(neuron_ids)}')
            # ref: https://medium.com/@hunter-j-phillips/a-simple-introduction-to-dropout-3fd41916aaea
            hooks.append(mediators[component](int(layer_id)).register_forward_hook(prunning_neurons(neuron_ids, nie_m, multiplier, param_name, DEBUG)))

    return model, hooks

def compute_maf1(raw_distribution_path, label_maps):
    label_remaps = {v:k for k, v in label_maps.items()}
    with open(raw_distribution_path, 'rb') as handle: 
        distributions = pickle.load(handle)
        golden_answers = pickle.load(handle)
        print(f'loading distributions and labels from : {raw_distribution_path}')
    maf1 = {} 
    if isinstance(distributions, dict):
        modes = list(distributions.keys())
    else:
        modes = ['Null']
        distributions  = {'Null': distributions}
        golden_answers = {'Null': golden_answers}
    
    print(f'compute_maf1 modes:{modes}') # Intervene, Null
     
    for mode in modes:
        maf1[mode] = {k: [] for k in (['all'] + list(label_maps.keys()))}
        y_pred = [int(torch.argmax(y)) for y in distributions[mode]]
        y_true = [int(y) for y in golden_answers[mode]]
        maf1[mode]['all'] = f1_score(y_true, y_pred, average='macro')
        for value in label_maps.values():
            maf1[mode][label_remaps[value]] = f1_score(y_true, y_pred, labels=[0,1],average=None)[value]
            maf1[mode][label_remaps[value]] = f1_score(y_true, y_pred, labels=[0,1],average=None)[value]
        # to_pop = []
        # for key in maf1[mode].keys():
        #     if len(maf1[mode][key])==0:
        #         to_pop.append(key)
        # for p in to_pop:
        #     maf1[mode].pop(p, None)
    return maf1    