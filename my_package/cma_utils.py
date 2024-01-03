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
from my_package.utils import get_outliers
from my_package.counter import count_negations 

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

def get_component_names(config):

    representations = {}
    layers = config["layers"]
    is_averaged_embeddings = config["is_averaged_embeddings"]
    is_group_by_class = config["is_group_by_class"]

    for component in (["Q","K","V","AO","I","O"]):
        representations[component] = {}
        for do in [config["treatment"]]:
            representations[component][do] = {}
            for layer in layers:
                if config["is_averaged_embeddings"]:
                    representations[component][do][layer] = []
                elif config["is_group_by_class"]:
                    representations[component][do][layer] = {}
                    for label_text in config['label_maps'].keys(): 
                        representations[component][do][layer][label_text] = []
                    
    
    print(representations)

    return representations

def foward_hooks(do, tokenizer, dataloader, model, DEVICE, class_name=None, DEBUG=0):
    """  foward to hook all representations """
    counter = 0
    
    for batch_idx, (sentences, labels) in enumerate(tqdm(dataloader, desc=f"counterfactual_set_loader")):
        if DEBUG >=4: print(f' ************** batch_idx: {batch_idx} ***************')
        # ******************* hooks representations **********************
        premise, hypo = sentences[do] if class_name is None else sentences[do][class_name]
        pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()} 

        with torch.no_grad():    
            outputs = model(**inputs)
        
        del outputs
        counter += inputs['input_ids'].shape[0]
        inputs = {k: v.to('cpu') for k,v in inputs.items()} 
    
    return counter


def register_hooks(hooks, do, layer_modules, config, representations, class_name=None):
      
    layers = config["layers"] 
    is_averaged_embeddings = config["is_averaged_embeddings"]
    is_group_by_class = config["is_group_by_class"]

    DEBUG = config["DEBUG"]

    for component in (["Q","K","V","AO","I","O"]):
        for layer in layers:
            hooks.append(layer_modules[component](layer).register_forward_hook(get_activation(layer, do, component, representations, is_averaged_embeddings, class_name=class_name, DEBUG=DEBUG)))
    
    return hooks

def check_counterfactuals(representations, config, counter):
    for component in (["Q","K","V","AO","I","O"]):
        for do in [config["treatment"]]:
            for layer in config["layers"]:
                if config["is_averaged_embeddings"]:
                    assert len(representations[component][do][layer]) == counter, f"Expected : {counter}, but found on layer{layer}, {component}, {do} :{len(representations[component][do][layer])}"
                elif config["is_group_by_class"]:
                    for label_text in config['label_maps'].keys(): 
                        assert len(representations[component][do][layer][label_text]) == counter, f"Expected : {counter}, but found on {component},{do}, layer{layer},{label_text}:{len(representations[component][do][layer][label_text])}"

    print(f'Counterfactuals pass test !')

def collect_counterfactuals(model, model_path, dataset_name, method_name, seed,  counterfactual_paths, config, experiment_set, dataloader, tokenizer, DEVICE, all_seeds=False): 
    """ getting all activation's neurons used as mediators(Z) to compute NIE scores later """
    from my_package.utils import load_model
    import copy
    
    if model_path is not None: 
        _model = load_model(path= model_path, model=copy.deepcopy(model))
        print(f'Loading Counterfactual model: {model_path}')
    else:
        _model = copy.deepcopy(model)
        _model = _model.to(DEVICE)
        print(f'using original model as input to this function')
    
    layers = config["layers"] 
    is_averaged_embeddings = config["is_averaged_embeddings"]
    # getting counterfactual of all components(eg. Q, K) for specific seed
    _counterfactual_paths = counterfactual_paths
    
    #dicts to store the activations
    representations = {}
    # dict to store  probabilities
    distributions = {}
    counter = 0

    layer_modules = get_mediators(_model)
    representations = get_component_names(config)
    treatments = [config["treatment"]]

    for do in treatments:
        if config["is_averaged_embeddings"]:
            hooks = []
            hooks = register_hooks(hooks, do, layer_modules, config, representations, class_name=None)
            counter = foward_hooks(do, tokenizer, dataloader, _model, DEVICE, DEBUG=config['DEBUG'])
            for hook in hooks: hook.remove()
            del hooks
        elif config["is_group_by_class"]:
            for text_label in config['label_maps'].keys():
                hooks = []
                hooks = register_hooks(hooks, do, layer_modules, config, representations, class_name=text_label)
                counter = foward_hooks(do, tokenizer, dataloader, _model, DEVICE, class_name=text_label, DEBUG=config['DEBUG'])
                for hook in hooks: hook.remove()
                del hooks
    
    check_counterfactuals(representations, config, counter)
    
    for cur_path in _counterfactual_paths:
        component = cur_path.split('/')[-1].split('_')[1 if is_averaged_embeddings else 3]
        print(f'{component}, :{cur_path}')
        
        with open(cur_path, 'wb') as handle: 
            pickle.dump(representations[component], handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"saving counterfactual and counter into {cur_path} done ! ")
                
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

def geting_counterfactual_paths(config, method_name, seed=None):

    path = f'../counterfactuals/{method_name}/'
    if not os.path.exists(path): os.mkdir(path) 
    path = os.path.join(path, "seed_"+ str( config['seed'] if seed is None else seed ) ) 
    if not os.path.exists(path): os.mkdir(path) 

    counterfactual_paths = []
    is_counterfactual_exist = []
    
    for component in tqdm(["Q","K","V","AO","I","O"], desc="Components"): 
        if config["is_averaged_embeddings"]:
            if config["is_group_by_class"]:
                cur_path = f'avg_class_level_{component}_counterfactual_representation.pickle'
            else:
                cur_path = f'avg_{component}_counterfactual_representation.pickle'
        else:
            if config["is_group_by_class"]:
                if component == "I":
                    for  do in ['High-overlap','Low-overlap']:
                        for class_name in ["contradiction","entailment","neutral"]:
                            cur_path = f'individual_class_level_{component}_{do}_{class_name}_counterfactual_representation.pickle'
                            counterfactual_paths.append(os.path.join(path, cur_path))
                            is_counterfactual_exist.append(os.path.isfile(os.path.join(path, cur_path)))
                else: 
                    cur_path = f'individual_class_level_{component}_counterfactual_representation.pickle'
                    counterfactual_paths.append(os.path.join(path, cur_path))
                    is_counterfactual_exist.append(os.path.isfile(os.path.join(path, cur_path)))
                continue
            else:
                cur_path = f'individual_{component}_counterfactual_representation.pickle'

        counterfactual_paths.append(os.path.join(path, cur_path))
        is_counterfactual_exist.append(os.path.isfile(os.path.join(path, cur_path)))

    return counterfactual_paths, is_counterfactual_exist

def get_overlap_thresholds(df, upper_bound, lower_bound, dataset_name):
     
    thresholds = {"High-overlap": None, "Low-overlap": None}
    
    if dataset_name == 'fever':
        df['count_negations']  = df['claim'].apply(count_negations)
    elif dataset_name == 'qqp':
        df['overlap_scores'] = df['pair_label'].apply(get_qqp_bias_scores)
        # df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)
    else:    
        df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)
        
    biased_scores = df['count_negations'] if dataset_name == 'fever'  else df['overlap_scores'] 

    thresholds["Low-overlap"]  = 0.0 if dataset_name == 'fever' else np.percentile(biased_scores, lower_bound)
    thresholds["High-overlap"] = 1.0 if dataset_name == 'fever' else np.percentile(biased_scores, upper_bound)

    return thresholds
    
def group_by_treatment(thresholds, overlap_score, gold_label):

    # note : we dont care about 
    if overlap_score >= thresholds["High-overlap"]:
        return "HOL"
    elif overlap_score <= thresholds["Low-overlap"]: 
        return "LOL"
    else:
        return "exclude"

def get_activation(layer, do, component, activation, is_averaged_embeddings, class_name = None, DEBUG=0):
  # the hook signature
  def hook(model, input, output):
    # bz, seq_len, hid_dim
    if class_name is None and is_averaged_embeddings:
        activation[component][do][layer].extend(output.detach()[:,0,:])
    else: 
        activation[component][do][layer][class_name].extend(output.detach()[:,0,:])

    if DEBUG >=4: print(f"{do}, layer : {layer}, component: {component}, class_name: {class_name},inp:{input[0].shape}, out:{output.shape} ")
  
  return hook

def get_overlap_score(pair_label):
    prem_words = []
    hyp_words = []

    sentence1 = pair_label[0].strip()
    sentence2 = pair_label[1].strip()
    gold_label = pair_label[2].strip() 

    for word in sentence1.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in sentence2.split():
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

def get_qqp_bias_scores(pair_label):
    # order of word are not import important  for pharaphasing 
    sent1 = []
    sent2 = []

    sentence1 = pair_label[0].strip()
    sentence2 = pair_label[1].strip()
    gold_label = pair_label[2].strip() 

    sentence1 = set(sentence1.split())
    sentence2 = set(sentence2.split())

    for ignore  in [".", "?", "!"]:
        if ignore in sentence1: sentence1.remove(ignore)
        if ignore in sentence2: sentence2.remove(ignore)

    if sentence1 == sentence2:
        return 2.0
    else:
        sent = {}
        major_sent = sentence1 if len(sentence1) > len(sentence2) else sentence2
        minor_sent = sentence1 if len(sentence1) < len(sentence2) else sentence2
    
        count = 0
        for word in minor_sent:
            if word in major_sent:
                count+=1
    
        overlap_score = count/len(minor_sent)        

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
                        intervention_type, 
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
    
    ret = 0
    counter = 0
    ratio = 0
    class_counters = {}
    class_ratios = {}
    delta_y = {}


    # all collectors
    for mode in interventions:

        distributions[mode] = {}
        for label in list(label_maps.keys()):
            distributions[mode][label] = []
            if label not in class_counters.keys(): class_counters[label] = 0
            if label not in class_ratios.keys(): class_ratios[label] = []
            if label not in delta_y.keys(): delta_y[label] = []
    
     # mediator used to intervene
    mediators["Q"] = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    mediators["K"] = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    mediators["V"] = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    mediators["AO"]  = lambda layer : model.bert.encoder.layer[layer].attention.output
    mediators["I"]  = lambda layer : model.bert.encoder.layer[layer].intermediate
    mediators["O"]  = lambda layer : model.bert.encoder.layer[layer].output

    cls = get_hidden_representations(counterfactual_paths, layers, is_group_by_class, is_averaged_embeddings)
    
    # why Z value is not the same as trace counterfactual ?
    Z = cls[component][do][layer]

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
                hooks.append(mediators[component](layer).register_forward_hook(neuron_intervention(neuron_ids = [neuron_id], 
                                                                                                  DEVICE = DEVICE ,
                                                                                                  value = Z, 
                                                                                                  intervention_type=intervention_type)))

            with torch.no_grad(): 
                # Todo: generalize to distribution if the storage is enough
                cur_dist[mode] = F.softmax(model(**inputs).logits , dim=-1)
            # if debug: print(f"+++++++++++++ batch_idx: {batch_idx}, mode {mode} ++++++++")
            # if debug: print(cur_dist)
            if mode == "Intervene": 
                for hook in hooks: hook.remove() 

        cur_ret = cur_dist["Intervene"][:,label_maps["entailment"]] / cur_dist["Null"][:,label_maps["entailment"]] 

        # quanitifying the impacts 
        del_ret = cur_dist['Intervene'] - cur_dist['Null']
        cur_ratio = cur_dist["Intervene"] / cur_dist["Null"]

        ratio += torch.sum(cur_ratio, dim=0)
        
        ret += torch.sum(cur_ret - 1, dim=0)
        counter += inputs['input_ids'].shape[0]
        #ret = ret / inputs['input_ids'].shape[0]

        # if debug: print(f'batch_idx : {batch_idx}, ratio : {ret - 1}') 

        for sample_idx in range(cur_ratio.shape[0]):
            
            class_counters[labels[sample_idx]] += 1 
            class_ratios[labels[sample_idx]].append(cur_ratio[sample_idx,:])  
            delta_y[labels[sample_idx]].append(del_ret[sample_idx,:])

            for mode in interventions: distributions[mode][labels[sample_idx]].append(cur_dist[mode][sample_idx,:])

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
    # every sets dont follow normal distribution 

    mean = {}
    median = {}
    outliers = {}

    # set group by golden 
    for golden in ['contradiction','entailment','neutral']:
        class_ratios[golden] = torch.stack(class_ratios[golden],dim=0).cpu()
        delta_y[golden] = torch.stack(delta_y[golden],dim=0).cpu()
        median[golden] = torch.median(class_ratios[golden], dim=0)[0]
        mean[golden] = torch.mean(class_ratios[golden], dim=0)

        if debug:
            print(f"++++++++++++++++  current {golden}  set +++++++++++++++++++++") 
            print(f">>>>>>>>>>>>> Y_intervene / Y_Null ratios <<<<<<<<<<<<<")
            print(f"Median {golden} set: {median[golden]}")
            print(f"Mean {golden} set: {mean[golden]}")

            print(f">>>>>>>>>>>> Y_intervene - Y_Null <<<<<<<<<<<<<<<<<<<< ")
            print(f"Median  {golden} set: {torch.median(delta_y[golden], dim=0)[0]}")
            print(f"Mean  {golden} set: {torch.mean(delta_y[golden], dim=0) }")
        # print(f"---------------------------------------------")

        outliers[golden] = []

        for type_output in ['contradiction','entailment','neutral']:
            if golden == type_output: continue
            if debug: print(f">>> current output prob:") 
            get_outliers(type_output, outliers[golden], label_maps, class_ratios[golden])
        for mode in interventions: 
            distributions[mode][golden] = torch.stack(distributions[mode][golden],dim=0)
    
    if not os.path.exists(dist_path):
        with open(dist_path, 'wb') as handle: 
            pickle.dump(class_ratios, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(median, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saving NIE scores into : {dist_path}')

def get_hidden_representations(config, counterfactual_paths, method_name, seed, layers, is_group_by_class, is_averaged_embeddings):
    
    if is_averaged_embeddings:
        # get average of [CLS] activations
        counterfactual_representations = {}
        avg_counterfactual_representations = {}
        for cur_path in counterfactual_paths:
            component = cur_path.split('/')[-1].split('_')[1]
            seed = cur_path.split('/')[3].split('_')[-1]
            if seed not in counterfactual_representations.keys(): counterfactual_representations[seed] = {}
            if seed not in avg_counterfactual_representations.keys(): avg_counterfactual_representations[seed] = {}
            avg_counterfactual_representations[seed][component] = {}
            # load all output components 
            with open(cur_path, 'rb') as handle:
                # get [CLS] activation [do][layer]
                counterfactual_representations[seed][component] = pickle.load(handle)
                # attention_data = pickle.load(handle)
                # counter = pickle.load(handle)
            treatments = ["High-overlap"]  if config['dataset_name'] == 'fever' else ["High-overlap", "Low-overlap"]
            for do in treatments:
                avg_counterfactual_representations[seed][component][do] = {}
                # concate all batches
                for layer in layers:
                    # compute average over samples
                    if is_group_by_class:
                        for class_name in counterfactual_representations[seed][component][do].keys():
                            if class_name not in avg_counterfactual_representations[seed][component][do].keys():
                                avg_counterfactual_representations[seed][component][do][class_name] = {}
                            avg_counterfactual_representations[seed][component][do][class_name][layer] = counterfactual_representations[seed][component][do][class_name][layer] / counter[do][class_name]
                    else:
                        avg_counterfactual_representations[seed][component][do][layer] = counterfactual_representations[seed][component][do][layer] / counter[do]
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

def geting_NIE_paths(config, method_name, mode, seed=None):
    NIE_paths = []
    is_NIE_exist = []
    path = f'../NIE/{method_name}/'
    if not os.path.exists(path): os.mkdir(path) 
    path = os.path.join(path, "seed_"+ str(config['seed'] if seed is None else seed ) )
    if not os.path.exists(path): os.mkdir(path) 
    layers = config['layers']  if config['computed_all_layers'] else [config['layer']]
    
    if config['is_averaged_embeddings']:
        if config['computed_all_layers']: 
            NIE_path = os.path.join(path, f'avg_embeddings_{mode[0]}_computed_all_layers_.pickle') 
            NIE_paths.append(NIE_path)
            is_NIE_exist.append(os.path.isfile(NIE_path))
        else:
            for layer in layers:
                # if not isinstance(layer, list): cur_layer = [layer]
                NIE_path = os.path.join(path, f'avg_embeddings_{mode[0]}_layer_{layer}_.pickle') 
                NIE_paths.append(NIE_path)
                is_NIE_exist.append(os.path.isfile(NIE_path))
    else:
        for cur_path in config['counterfactual_paths']:
            # extract infor from current path 
            component = cur_path.split('/')[-1].split('_')[1]
            class_name = None
            # NIE_path = os.path.join(path, f'avg_high_level_{layer}_{mode[0]}.pickle') 
            NIE_path = os.path.join(path, f'avg_embeddings_{mode[0]}_layer_{layer}_.pickle') 
            print(f"current path: {NIE_path} , is_exist : {os.path.isfile(cur_path)}")
            NIE_paths.append(NIE_path)
            is_NIE_exist.append(os.path.isfile(cur_path))
    return NIE_paths, is_NIE_exist

def get_nie_set_path(config, experiment_set, save_nie_set_path):
    """ prepare validation set used to compute NIE later"""
    combine_types = []
    pairs = {}

    if config['is_group_by_class']:
        nie_dataset = {}
        nie_loader = {}

        for type in config['label_maps'].keys():
            # get the whole set of validation 
            pairs[type] = list(experiment_set.df[experiment_set.df.gold_label == type].pair_label)
            # samples data (exclude mode samples)
            ids = list(torch.randint(0, len(pairs[type]), size=(config['num_samples'] //3,)))
            pairs[type] = np.array(pairs[type])[ids,:].tolist()
            nie_dataset[type] = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(pairs[type])]
            nie_loader[type] = DataLoader(nie_dataset[type], batch_size=32)

    else:
        # balacing nie set across classes
        for type in config['label_maps'].keys():
            # get the whole set of validation for each class
            pairs[type] = list(experiment_set.df[experiment_set.df.gold_label == type].pair_label)
            # samples data (exclude mode samples)
            ids = list(torch.randint(0, len(pairs[type]), size=(config['num_samples'] //3,)))
            pairs[type] = np.array(pairs[type])[ids,:].tolist()
            combine_types.extend(pairs[type])

        # dataset = [([premise, hypo], label) for idx, (premise, hypo, label) in enumerate(pairs['entailment'])]
        nie_dataset = [[[premise, hypo], label] for idx, (premise, hypo, label) in enumerate(combine_types)]
        nie_loader = DataLoader(nie_dataset, batch_size=32)
    
    with open(save_nie_set_path, 'wb') as handle:
        pickle.dump(nie_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(nie_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done saving validation set used to compute NIE into {save_nie_set_path} !") 

def summary_eval_counterfactual(average_all_seed_distributions, label_maps, all_paths):
    print('==== Summary ===')
    for do in ['High-overlap','Low-overlap']:
        print(f'>> {do}')
        for cur_class in label_maps.keys():
            print(f" {cur_class} acc : {average_all_seed_distributions[do][cur_class]/ len(all_paths)}")
