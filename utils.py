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
from intervention import neuron_intervention
from tabulate import tabulate
import statistics 


def report_gpu(): 
  print(f"++++++++++++++++++++++++++++++")
  print(f"before deleting : {torch.cuda.list_gpu_processes()}")
  gc.collect()
  torch.cuda.empty_cache()
  print(f"after emptying cache : {torch.cuda.list_gpu_processes()}")
  print(f"++++++++++++++++++++++++++++++")
  
def geting_counterfactual_paths(counterfactual_paths, is_counterfactual_exist, is_averaged_embeddings, is_group_by_class):


    for component in tqdm(["Q","K","V","AO","I","O"], desc="Components"): 

        if is_averaged_embeddings:
                
            if is_group_by_class:

                cur_path = f'../pickles/avg_class_level_{component}_counterfactual_representation.pickle'

            else:
                cur_path = f'../pickles/avg_{component}_counterfactual_representation.pickle'

        else:

            if is_group_by_class:
            
                    if component == "I":
                        
                        for  do in ['High-overlap','Low-overlap']:
                            
                            for class_name in ["contradiction","entailment","neutral"]:
                            
                                cur_path = f'../pickles/individual_class_level_{component}_{do}_{class_name}_counterfactual_representation.pickle'
                                
                                counterfactual_paths.append(cur_path)
                                is_counterfactual_exist.append(os.path.isfile(cur_path))
                                
                    else: 

                        cur_path = f'../pickles/individual_class_level_{component}_counterfactual_representation.pickle'

                        counterfactual_paths.append(cur_path)
                        is_counterfactual_exist.append(os.path.isfile(cur_path))

                    continue

            else:

                cur_path = f'../pickles/individual_{component}_counterfactual_representation.pickle'

        counterfactual_paths.append(cur_path)
        is_counterfactual_exist.append(os.path.isfile(cur_path))

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


def get_overlap_score(pair_label):

    prem_words = []
    hyp_words = []

    premise = pair_label[0].strip()
    hypothesis = pair_label[1].strip()
    gold_label = pair_label[2].strip()

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)

    count = 0
    for word in hyp_words:
        if word in prem_words:
            count+=1

    overlap_score = count/len(hyp_words)        

    return overlap_score

def trace_counterfactual(do, 
                        layer, 
                        model, 
                        save_nie_set_path, 
                        tokenizer,
                        DEVICE, 
                        layers, 
                        heads,
                        counterfactual_paths,
                        label_maps,
                        is_group_by_class, 
                        is_averaged_embeddings, 
                        debug = False):

    path = f'../pickles/top_neurons/top_neuron_{do}_{layer}.pickle'

    nie_dataloader = None
    hooks = None
    
    interventions = ["Null", "Intervene"]
    distributions = {} 
    mediators = {}

    with open(save_nie_set_path, 'rb') as handle:
        
         nie_dataset = pickle.load(handle)
         nie_dataloader = pickle.load(handle)
        
         print(f"loading nie sets from pickle {save_nie_set_path} !")  
  
    with open(path, 'rb') as handle:
        # get [CLS] activation 
        top_neuron = pickle.load(handle)
    
    
    component = list(top_neuron.keys())[0].split('-')[0]
    neuron_id  = int(list(top_neuron.keys())[0].split('-')[1])
    
    dist_path = f'../pickles/distributions/L{layer}_{do}_{component}_{neuron_id}.pickle'
    
    print(f"component : {component}")
    print(f"neuron_id : {neuron_id}")

    for mode in interventions: distributions[mode] = []
    
     # mediator used to intervene
    mediators["Q"] = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    mediators["K"] = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    mediators["V"] = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    mediators["AO"]  = lambda layer : model.bert.encoder.layer[layer].attention.output
    mediators["I"]  = lambda layer : model.bert.encoder.layer[layer].intermediate
    mediators["O"]  = lambda layer : model.bert.encoder.layer[layer].output

    cls = get_hidden_representations(counterfactual_paths, layers, heads, is_group_by_class, is_averaged_embeddings)
    
    Z = cls[component][do][layer]

    ret = 0
    counter = 0
    ratio = 0
    class_counters = {}
    class_ratios = {}

    for batch_idx, (sentences, labels) in enumerate(nie_dataloader):
        
        premise, hypo = sentences

        pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]
        
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
            
        cur_dist = {}

        print(f"batch_idx : {batch_idx}")

        for mode in interventions:


            if mode == "Intervene": 
                hooks = []
                hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(neuron_ids = [neuron_id], DEVICE = DEVICE ,value = Z)))

            with torch.no_grad(): 
                
                # Todo: generalize to distribution if the storage is enough
                cur_dist[mode] = F.softmax(model(**inputs).logits , dim=-1)
                distributions[mode].append(cur_dist[mode])
            
            # if debug: print(f"+++++++++++++ batch_idx: {batch_idx}, mode {mode} ++++++++")
            # if debug: print(cur_dist)
            
            if mode == "Intervene": 
                for hook in hooks: hook.remove() 

        cur_ret = cur_dist["Intervene"][:,label_maps["entailment"]] / cur_dist["Null"][:,label_maps["entailment"]] 
        cur_ratio = cur_dist["Intervene"] / cur_dist["Null"]
        ratio += torch.sum(cur_ratio, dim=0)
        
        ret += torch.sum(cur_ret - 1, dim=0)
        counter += inputs['input_ids'].shape[0]
        #ret = ret / inputs['input_ids'].shape[0]

        # if debug: print(f'batch_idx : {batch_idx}, ratio : {ret - 1}') 

        for sample_idx in range(cur_ratio.shape[0]):
            
            if labels[sample_idx] not in class_counters.keys(): 

                class_counters[labels[sample_idx]] = 0
                class_ratios[labels[sample_idx]] = []

            class_counters[labels[sample_idx]] += 1 
            # class_ratios[labels[sample_idx]] +=  cur_ratio[sample_idx,:]
            class_ratios[labels[sample_idx]].append(cur_ratio[sample_idx,:])  

        if debug: 

            for sample_idx in range(cur_dist["Intervene"].shape[0]):
                print(f"sample_idx : {sample_idx}, {labels[sample_idx]} sample")
                print(f'+------------------++----------------++--------------+')
                print(f'>>  contradiction    ||    entailment  ||     neutral  |')
                for mode in interventions:
                    print(f'{mode}   {cur_dist[mode][sample_idx,:]}')
                print(f'+------------------++----------------++--------------+')


    print(f"label_maps : {label_maps}") 
    print(f'NIE average : {ret/counter}')
    print(f"Average ratio for whole sets : {(ratio/counter).cpu().tolist()}")
    # print(f"aveaging ratio contradiction set: {(class_ratios['contradiction'] / class_counters['contradiction']).cpu().tolist()}")
    # print(f"averaging ratio entailment set: {(class_ratios['entailment']/ class_counters['entailment']).cpu().tolist()}")
    # print(f"averaging ratio neutral set: {(class_ratios['neutral'] / class_counters['neutral']).cpu().tolist()}")

    # every sets dont follow normal distribution 

    mean = {}

    for type in ['contradiction','entailment','neutral']:
        
        class_ratios[type] = torch.stack(class_ratios[type],dim=0).cpu()
        mean[type] = torch.median(class_ratios[type], dim=0)[0]
        print(f"Median ratio {type} set: {mean[type]}")

    breakpoint()
    # with open(NIE_path, 'wb') as handle: 
    #     pickle.dump(NIE, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(f'saving NIE scores into : {NIE_path}')


 
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

def get_overlap_thresholds(df, upper_bound, lower_bound):
    
    thresholds = {"High-overlap": None, "Low-overlap": None}

    df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)

    overlap_scores = df['overlap_scores'] 

    thresholds["Low-overlap"]  = np.percentile(overlap_scores, lower_bound)
    thresholds["High-overlap"] = np.percentile(overlap_scores, upper_bound)

    return thresholds
    
def group_by_treatment(thresholds, overlap_score, gold_label):

    # note : we dont care about 
    if overlap_score >= thresholds["High-overlap"]:
        return "HOL"
    elif overlap_score <= thresholds["Low-overlap"]: 
        return "LOL"
    else:
        return "exclude"

def get_activation(layer, do, component, activation, is_averaged_embeddings, class_name = None):

  # the hook signature
  def hook(model, input, output):
    
    # print(f"layer : {layer}, do : {do}, {output.shape} ")

    if class_name is None:
    
        if layer not in activation[component][do].keys():

            if is_averaged_embeddings:
                
                activation[component][do][layer] = 0

            else:
                activation[component][do][layer] = []

        # grab representation of [CLS] then sum up

        if is_averaged_embeddings:

            activation[component][do][layer] += torch.sum(output.detach()[:,0,:], dim=0)

        else:
            
            activation[component][do][layer].extend(output.detach()[:,0,:])
    else:

        if layer not in activation[component][do][class_name].keys():

            if is_averaged_embeddings:
                
                activation[component][do][class_name][layer] = 0

            else:
                activation[component][do][class_name][layer] = []

        # grab representation of [CLS] then sum up

        if is_averaged_embeddings:

            activation[component][do][class_name][layer] += torch.sum(output.detach()[:,0,:], dim=0)

        else:
            activation[component][do][class_name][layer].extend(output.detach()[:,0,:])
  
  return hook

def collect_output_components(model, counterfactual_paths, experiment_set, dataloader, tokenizer, DEVICE, layers, heads, is_averaged_embeddings):
    
    """ getting all neurons used as mediators(Z) later """
   
    layer_modules = {}

    # using for register
    registers = None

    #dicts to store the activations
    q_activation = {}
    k_activation = {}
    v_activation = {}
  
    ao_activation = {}
    intermediate_activation = {}
    out_activation = {} 

    hidden_representations = {}

    
    attention_data = {}        

    # dict to store  probabilities
    distributions = {}
    
    counter = {}

    batch_idx = 0
    
    hooks =  {"High-overlap" : None, "Low-overlap": None}

    # linear layer
    layer_modules["Q"] = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    layer_modules["K"] = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    layer_modules["V"] = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    layer_modules["AO"] = lambda layer : model.bert.encoder.layer[layer].attention.output
    layer_modules["I"] = lambda layer : model.bert.encoder.layer[layer].intermediate
    layer_modules["O"] = lambda layer : model.bert.encoder.layer[layer].output

    for component in (["Q","K","V","AO","I","O"]):
        
        hidden_representations[component] = {}
    
    # **** collecting all counterfactual representations ****    
    for batch_idx, (sentences, labels) in enumerate(tqdm(dataloader, desc=f"Intervene_set_loader")):
        
        for idx, do in enumerate(tqdm(['High-overlap','Low-overlap'], desc="Do-overlap")):
            
            if do not in hidden_representations[component].keys():

                for component in (["Q","K","V","AO","I","O"]):

                    hidden_representations[component][do] = {}
                
                distributions[do] = {} 
                counter[do] = {} if experiment_set.is_group_by_class else 0

            if experiment_set.is_group_by_class:

                for class_name in sentences[do].keys():

                    registers = {}

                    # register all modules
                    if class_name not in counter[do].keys():
                                
                        counter[do][class_name] = 0 

                    for component in (["Q","K","V","AO","I","O"]):
                    
                        if class_name not in hidden_representations[component][do].keys():

                                hidden_representations[component][do][class_name] = {}

                                # distributions[do][class_name] = {} 

                        registers[component] = {}
                    
                        for layer in layers:

                            if layer not in hidden_representations[component][do][class_name].keys():
                                
                                hidden_representations[component][do][class_name][layer] = []
                            
                            registers[component][layer] = layer_modules[component](layer).register_forward_hook(get_activation(layer, do, component, hidden_representations, is_averaged_embeddings, class_name=class_name))                        

                    premise, hypo = sentences[do][class_name]
    
                    pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
    
                    inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")

                    inputs = {k: v.to(DEVICE) for k,v in inputs.items()} 

                    counter[do][class_name] += inputs['input_ids'].shape[0]
            
                    with torch.no_grad():    

                        outputs = model(**inputs)

                    # detach the hooks
                    for layer in layers:

                        for component in ["Q","K","V","AO","I","O"]:

                            registers[component][layer].remove()

            else:
                
                registers = {}
                 
                # register all modules
                for component in (["Q","K","V","AO","I","O"]):
                        
                    registers[component] = {}
                    
                    for layer in layers:

                        registers[component][layer] = layer_modules[component](layer).register_forward_hook(get_activation(layer, do, component, hidden_representations, is_averaged_embeddings))                        
                    
                # forward to collect counterfactual representations
                premise, hypo = sentences[do]

                pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]

                inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")

                inputs = {k: v.to(DEVICE) for k,v in inputs.items()} 

                counter[do] += inputs['input_ids'].shape[0]
        
                with torch.no_grad():    

                    outputs = model(**inputs)

            del outputs
            
            inputs = {k: v.to('cpu') for k,v in inputs.items()} 

        
        batch_idx += 1
   
    # **** Writing the all counterfactual representations into pickles ****
    for cur_path in counterfactual_paths:

        component = sorted(cur_path.split("_"), key=len)[0]  
        
        if component == "I" and not is_averaged_embeddings:

            do = cur_path.split("_")[4]
            class_name = cur_path.split("_")[5]

            # hidden_representations[component][do][class_name][layer][sample_idx]
            with open(cur_path,'wb') as handle: 
                pickle.dump(hidden_representations[component][do][class_name], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"saving to {cur_path} done ! ")
        else:

            with open(cur_path, 'wb') as handle: 
                
                # nested dict : [component][do][class_name][layer][sample_idx]
                pickle.dump(hidden_representations[component], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"saving to {cur_path} done ! ")
                

    with open('../pickles/utilizer_components.pickle', 'wb') as handle: 
        
        pickle.dump(attention_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(experiment_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"save utilizer to ../pickles/utilizer_components.pickle  ! ")

def new_features(v):
    print(v)
    return v
    
def test_mask(neuron_candidates =[]):

    x  = torch.tensor([[ [1,2,3], 
                         [4,5,6]
                       ],
                       [
                         [1,2,3],
                         [4,5,6]                           
                       ],
                       [
                         [1,2,3],
                         [4,5,6]                           
                       
                       ]])

    neuron_ids = [0, 1, 2]

    # define used to intervene
    mask = torch.zeros_like(x, dtype= torch.bool) 
    
    # bz, seq_len, hidden_dim
    mask[:, 0, neuron_ids] = 1

    print(f"Before masking X")
    print(x)

    print(f" ===  mask list of neurons : {neuron_ids}  and [CLS] ====")
    print(mask)

    value = torch.tensor([11, 12, 13])[neuron_ids]
    value = value.repeat(x.shape[0], x.shape[1], 1)
    
    print(f"after masking X ")
    print(x.masked_scatter_(mask, value))

def get_hidden_representations(counterfactual_paths, layers, heads, is_group_by_class, is_averaged_embeddings):
        
    paths = { k : v for k, v in zip(["Q","K","V","AO","I","O"], counterfactual_paths)}
    
    with open('../pickles/utilizer_components.pickle', 'rb') as handle: 
        attention_data = pickle.load(handle)
        counter = pickle.load(handle)
        # experiment_set = pickle.load(handle)
        # dataloader, handle = pickle.load(handle)

    if is_averaged_embeddings:

        # get average of [CLS] activations
        counterfactual_representations = {}
        avg_counterfactual_representations = {}

        for component, cur_path in paths.items():

            avg_counterfactual_representations[component] = {}

            # load all output components 
            with open(cur_path, 'rb') as handle:
                
                # get [CLS] activation 
                counterfactual_representations[component] = pickle.load(handle)
                # attention_data = pickle.load(handle)
                # counter = pickle.load(handle)
            for do in ["High-overlap", "Low-overlap"]:
            
                avg_counterfactual_representations[component][do] = {}
                
                # concate all batches
                for layer in layers:

                    # compute average over samples
                    if is_group_by_class:
                    
                        for class_name in counterfactual_representations[component][do].keys():

                            if class_name not in avg_counterfactual_representations[component][do].keys():

                                avg_counterfactual_representations[component][do][class_name] = {}
                            
                            avg_counterfactual_representations[component][do][class_name][layer] = counterfactual_representations[component][do][class_name][layer] / counter[do][class_name]

                    else:
                            avg_counterfactual_representations[component][do][layer] = counterfactual_representations[component][do][layer] / counter[do]

        return  avg_counterfactual_representations

def get_single_representation(cur_path, do = None, class_name = None):
    
    component = sorted(cur_path.split("_"), key=len)[0]  
    
    hidden_representations = {}
    
    if component == "I":

        hidden_representations[component] = {}
        hidden_representations[component][do] = {}
        
        """
        saving to ../pickles/individual_class_level_Q_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_K_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_V_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_AO_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_I_High-overlap_contradiction_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_I_High-overlap_entailment_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_I_High-overlap_neutral_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_I_Low-overlap_contradiction_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_I_Low-overlap_entailment_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_I_Low-overlap_neutral_counterfactual_representation.pickle done ! 
        saving to ../pickles/individual_class_level_O_counterfactual_representation.pickle done ! 
        """
        
        cur_path = f'../pickles/individual_class_level_{component}_{do}_{class_name}_counterfactual_representation.pickle'
        
        # nested dict : [component][do][class_name][layer][sample_idx]
        with open(cur_path, 'rb') as handle:
            hidden_representations[component][do][class_name] = pickle.load(handle)
            print(f"loading from pickle {cur_path} !")        
    
    else:

        with open(cur_path, 'rb') as handle:
            hidden_representations[component] = pickle.load(handle)
            print(f"loading from pickle {cur_path} !")        

    return hidden_representations


class Classifier(nn.Module):

    def __init__(self, model):
        super(Classifier, self).__init__()

        self.model = model

        self.pooler = model.bert.pooler
        self.dropout = self.model.dropout
        self.classifier = self.model.classifier

    def forward(self, last_hidden_state):

        pooled_output = self.pooler(last_hidden_state)
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)

        return logits



def geting_NIE_paths(NIE_paths, layer, do, counterfactual_paths, is_NIE_exist, is_averaged_embeddings, is_group_by_class):

    if is_averaged_embeddings:

        NIE_path = f'../pickles/NIE/NIE_avg_high_level_{layer}_{do[0]}.pickle'
        NIE_paths.append(NIE_path)
        is_NIE_exist.append(os.path.isfile(NIE_path))

    else:
    
        for cur_path in counterfactual_paths:
            
            # extract infor from current path 
            component = sorted(cur_path.split("_"), key=len)[0]  
            class_name = None
            
            NIE_path = f'../pickles/NIE/NIE_avg_high_level_{layer}_{do[0]}.pickle'
            
            print(f"current path: {NIE_path} , is_exist : {os.path.isfile(cur_path)}")

            NIE_paths.append(NIE_path)
            is_NIE_exist.append(os.path.isfile(cur_path))
