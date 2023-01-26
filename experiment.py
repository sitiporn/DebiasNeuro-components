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
from utils import Intervention, get_overlap_thresholds, group_by_treatment, test_mask
from utils import collect_output_components
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)

# from fairseq.data.data_utils import collate_tokens
# from fairseq.models.roberta import RobertaModel


class ExperimentDataset(Dataset):
    def __init__(self, data_path, json_file, upper_bound, lower_bound, encode, DEBUG=False) -> None:

        data_path = os.path.join(data_path, json_file)

        self.df = pd.read_json(data_path, lines=True)

        self.encode = encode

        self.premises = {}
        self.hypothesises = {}
        self.labels = {}
        self.intervention = {}
        pair_and_label = []

        if DEBUG: print(self.df.columns)

        for i in range(len(self.df)):
            pair_and_label.append(
                (self.df['sentence1'][i], self.df['sentence2'][i], self.df['gold_label'][i]))

        self.df['pair_label'] = pair_and_label
        thresholds = get_overlap_thresholds(self.df, upper_bound, lower_bound)

        # get w/o treatment on entailment set
        self.df['is_treatment'] = self.df.apply(lambda row: group_by_treatment(
            thresholds, row.overlap_scores, row.gold_label), axis=1)

        self.df_exp_set = {"do-treatment": self.get_intervention_set(),
                        "no-treatment": self.get_base_set()}

        # Randomized Controlled Trials (RCTs)
        self.nums = {}

        self.nums['do-treatment'] = len(self.df_exp_set["do-treatment"])
        self.nums['no-treatment'] = len(self.df_exp_set["no-treatment"])

        self.equal_number = self.nums['do-treatment'] if self.nums['do-treatment'] <  self.nums['no-treatment'] else self.nums['no-treatment'] 

        self.df_exp_set["do-treatment"]  = self.df_exp_set['do-treatment'].reset_index(drop=True)
        self.df_exp_set["no-treatment"] = self.df_exp_set['no-treatment'].reset_index(drop=True)

        indexes = [*range(0, self.equal_number, 1)]
        sampling_idxes = []

        # sampling
        for  i in range(self.equal_number):
            idx = random.choice(indexes)

            while idx in sampling_idxes:
                idx = random.choice(indexes)

            sampling_idxes.append(idx)

        assert len(set(sampling_idxes)) == self.equal_number
        
        if self.nums['do-treatment'] > self.equal_number:

            self.df_exp_set["do-treatment"] = self.df_exp_set['do-treatment'].loc[sampling_idxes] 
            self.df_exp_set["do-treatment"]  = self.df_exp_set['do-treatment'].reset_index(drop=True)
        else:

            self.df_exp_set["no-treatment"] = self.df_exp_set['no-treatment'].loc[sampling_idxes]
            self.df_exp_set["no-treatment"] = self.df_exp_set['no-treatment'].reset_index(drop=True)

        

        for op in ["do-treatment", "no-treatment"]:

            self.premises[op] = list(self.df_exp_set[op].sentence1)
            self.hypothesises[op] = list(self.df_exp_set[op].sentence2)
            self.labels[op] = list(self.df_exp_set[op].gold_label)


            self.intervention[op] = Intervention(encode = self.encode,
                                    premises = self.premises[op],
                                    hypothesises = self.hypothesises[op]
                                    )

        # word overlap  more than 80 percent
        # self.premises["do-treatment"] = list(intervention_set.sentence1)
        # self.hypothesises["do-treatment"] = list(intervention_set.sentence2)

        # word overlap less than 20 percent
        # self.premises["no-treatment"] = list(base_set.sentence1)
        # self.hypothesises["no-treatment"] = list(base_set.sentence2)
    
    def get_intervention_set(self):

        # get high overlap score pairs
        return self.df[self.df['is_treatment'] == "treatment"]

    def get_base_set(self):

        # get low overlap score pairs
        return self.df[self.df['is_treatment'] == "no-treatment"]

    def __len__(self):
        # Todo: generalize label
        return self.equal_number
        # len(self.intervention.pair_sentences)
    
    def __getitem__(self, idx):

        pair_sentences = {}
        labels = {}

        pair_sentences['do-treatment'] = self.intervention['do-treatment'].pair_sentences[idx]
        pair_sentences['no-treatment'] = self.intervention['no-treatment'].pair_sentences[idx]

        labels['do-treatment'] = self.labels['do-treatment'][idx]
        labels['no-treatment'] = self.labels['no-treatment'][idx]
        
        return pair_sentences , labels


def prunning(model, layers):

    # Todo: generalize for any models
    # ref- https://arxiv.org/pdf/2210.16079.pdf
    # Todo: get Wl Wl_K , Wl_Q, Wl_V , Wl_AO, Wl_I , Wl_O of layer
    Wl_Q = lambda layer : model.bert.encoder.layer[layer].attention.self.query.weight.data
    Wl_K = lambda layer : model.bert.encoder.layer[layer].attention.self.key.weight.data
    Wl_V = lambda layer : model.bert.encoder.layer[layer].attention.self.value.weight.data
    Wl_AO = lambda layer : model.bert.encoder.layer[layer].output.dense.weight.data
    Wl_I  = lambda layer : model.bert.encoder.layer[layer].intermediate.dense.weight.data
    Wl_O =  lambda layer : model.bert.encoder.layer[layer].output.dense.weight.data

    for layer in layers:
        # inital all mask are value
        Ml_Q = torch.zeros_like(Wl_Q(layer))
        Ml_K = torch.zeros_like(Wl_K(layer))
        Ml_V = torch.zeros_like(Wl_V(layer))
        Ml_AO = torch.zeros_like(Wl_AO(layer))
        Ml_I  = torch.zeros_like(Wl_I(layer))
        Ml_O = torch.zeros_like(Wl_O(layer))

        # Todo: change to data.copy mode
        with torch.no_grad(): 
            model.bert.encoder.layer[layer].attention.self.query.weight.data.copy_(Wl_Q(layer) *  Ml_Q )
            # model.bert.encoder.layer[layer].attention.self.key.weight = Wl_K(layer) *  Ml_K 
            # model.bert.encoder.layer[layer].attention.self.key.value = Wl_V(layer) *  Ml_V 
            # model.bert.encoder.layer[layer].output.dense.weight = Wl_AO(layer) *  Ml_AO
            # model.bert.encoder.layer[layer].intermediate.dense = Wl_I(layer) *  Ml_I 
            # model.bert.encoder.layer[layer].output.dense = Wl_O(layer) *  Ml_O 

def get_average_activations(path, layers, heads):

    # load all output components 
    with open(path, 'rb') as handle:
        
        q_activation = pickle.load(handle)
        k_activation = pickle.load(handle)
        v_activation = pickle.load(handle)
        
        ao_activation = pickle.load(handle)
        intermediate_activation = pickle.load(handle)
        out_activation = pickle.load(handle)
        
        attention_data = pickle.load(handle)

    q_cls  = {}
    k_cls  = {}
    v_cls  = {}
    
    ao_cls = {}
    intermediate_cls = {}
    out_cls = {}
    attention_cls = {}

    # Todo: 
    # 1. get average over [CLS] of High overlap; do-treatment
    # 2. get average over [CLS] of High overlap; no-treatment

    # Todo: loop over [CLS] token across batches 
    # compute average of [CLS]


    for batch_idx in range(len(attention_data['do-treatment'])):

        for do in ['do-treatment','no-treatment']:

            if do not in ao_cls.keys():

                q_cls[do] = {}
                k_cls[do] = {}
                v_cls[do] = {} 
                
                ao_cls[do] = {}
                intermediate_cls[do] = {}
                out_cls[do] = {}
                # attention_cls[do] = {}

            for layer in layers:

                if  layer not in ao_cls[do].keys():

                    q_cls[do][layer] = []
                    k_cls[do][layer] = []
                    v_cls[do][layer] = []                   
                    
                    ao_cls[do][layer] = []
                    intermediate_cls[do][layer] = []
                    out_cls[do][layer] = []

                # grab [CLS]
                # batch_size, seq_len, hidden_dim
                q_cls[do][layer].append(q_activation[do][layer][batch_idx][:, 0, :])
                k_cls[do][layer].append(k_activation[do][layer][batch_idx][:, 0, :])
                v_cls[do][layer].append(v_activation[do][layer][batch_idx][:, 0, :])

                ao_cls[do][layer].append(ao_activation[do][layer][batch_idx][:, 0, :])
                intermediate_cls[do][layer].append(intermediate_activation[do][layer][batch_idx][:, 0, :])
                out_cls[do][layer].append(out_activation[do][layer][batch_idx][:, 0, :])

                #  in heads:
                #     # batch_size, num_heads, seq_len, seq_len
                #     attention_data['do-treatment'][batch_idx][layer][heads] 
    
    for do in ['do-treatment','no-treatment']:
         
         # concate all batches
        for layer in layers:

            # convert list to tensor
            q_cls[do][layer] = torch.cat(q_cls[do][layer], dim=0)    
            k_cls[do][layer] = torch.cat(k_cls[do][layer], dim=0)    
            v_cls[do][layer] = torch.cat(v_cls[do][layer], dim=0)    

            ao_cls[do][layer] = torch.cat(ao_cls[do][layer], dim=0)    
            intermediate_cls[do][layer] = torch.cat(intermediate_cls[do][layer], dim=0)    
            out_cls[do][layer] = torch.cat(out_cls[do][layer], dim=0)    

            # compute average over samples
            q_cls[do][layer] = torch.mean(q_cls[do][layer], dim=0)
            k_cls[do][layer] = torch.mean(k_cls[do][layer], dim=0)
            v_cls[do][layer] = torch.mean(v_cls[do][layer], dim=0)

            ao_cls[do][layer] = torch.mean(ao_cls[do][layer], dim=0)
            intermediate_cls[do][layer] = torch.mean(intermediate_cls[do][layer] , dim=0)
            out_cls[do][layer] = torch.mean(out_cls[do][layer], dim=0)

    return q_cls, k_cls, v_cls, ao_cls, intermediate_cls,  out_cls 

def neuron_intervention(neuron_ids, 
                       value,
                       intervention_type='replace'):
    
    # Hook for changing representation during forward pass
    def intervention_hook(module,
                            input,
                            output):
        
        # define mask where to overwrite
        scatter_mask = torch.zeros_like(output, dtype = torch.bool)

        # where to intervene
        # bz, seq_len, hidden_dim
        scatter_mask[:,0, neuron_ids] = 1
        
        neuron_values = value[neuron_ids]

        neuron_values = neuron_values.repeat(output.shape[0], output.shape[1], 1)

        output.masked_scatter_(scatter_mask, neuron_values)

    return intervention_hook

def cma_analysis(path, model, layers, heads, tokenizer, experiment_set, DEVICE):

    q_cls_avg, k_cls_avg, v_cls_avg, ao_cls_avg, intermediate_cls_avg,  out_cls_avg = get_average_activations(path, layers, heads)

    breakpoint()
    # For every samples
    # E(Yi|Xi=1)−E(Yi|Xi=0) 

    # HOL and LOL

    # guess prediction
    # for pair_sentences , labels in tqdm(dataloader):

    # Todo get pair sentences of all overlap scores
    pair_labels = list(experiment_set.df.pair_label)
    dataset = [([premise, hypo], label) for idx, (premise, hypo, label) in enumerate(pair_labels)]
    dataloader = DataLoader(dataset, batch_size=32)

    # self_attention = lambda layer : model.bert.encoder.layer[layer].attention.self
    self_output = lambda layer : model.bert.encoder.layer[layer].attention.output
    intermediate_layer = lambda layer : model.bert.encoder.layer[layer].intermediate
    output_layer = lambda layer : model.bert.encoder.layer[layer].output

    for i, (sentences, labels) in enumerate(dataloader):

        premise, hypo = sentences

        pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]
        
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits
        predictions = logits.argmax(dim=1)

        print(f"normal prediction : ")
        print(predictions)

        # run one full neuron intervention experiment
        for do in ['do-treatment','no-treatment']:
            for layer in layers:
                
                # get each prediction for nueron intervention
                # for neuron_id in range(ao_cls_avg[do][layer].shape[0]):
                #     # select layer to register  and input which neurons to intervene
                #     hooks = [] 


                #     neuron_ids = [*range(0, ao_cls_avg[do][layer].shape[0], 1)]

                #     # dont forget to change 11th to layer variable !
                #     hooks.append(self_output(layer).register_forward_hook(neuron_intervention(neuron_ids = neuron_ids, 
                #                                                               value = ao_cls_avg[do][11])))
                
                #     outputs = model(**inputs)
                #     logits = outputs.logits
                #     new_predictions = logits.argmax(dim=1)

                #     print(f"intervene prediction")
                #     print(predictions.eq(new_predictions))
                    
                #     for hook in hooks: hook.remove() 


                # for nueron_id in range(intermediate_cls_avg[do][layer].shape[0]):

                #     hooks = [] 
                #     hooks.append(intermediate_layer(layer).register_forward_hook(neuron_intervention()))

                #     outputs = model(**inputs)
                #     logits = outputs.logits
                #     predictions = logits.argmax(dim=1)
                    
                #     for hook in hooks: hook.remove() 
                
                for nueron_id in range(out_cls_avg[do][layer].shape[0]):
                    
                    hooks = [] 

                    neuron_ids = [*range(0, out_cls_avg[do][layer].shape[0], 1)]
                    
                    # dont forget to change 11th to layer variable !
                    hooks.append(output_layer(layer).register_forward_hook(neuron_intervention(neuron_ids = neuron_ids, 
                                                                    value = out_cls_avg[do][11])))
                        
                    outputs = model(**inputs)
                    logits = outputs.logits
                    new_predictions = logits.argmax(dim=1)

                    print(f"intervene prediction")
                    print(new_predictions)

                    print(f"== compare result ==")
                    print(predictions.eq(new_predictions))
                    print(f"# not change : {torch.sum(predictions.eq(new_predictions))}")
                    
                    for hook in hooks: hook.remove() 

                    breakpoint()


def main():

    # note: cannot concate because batching different seq_len
    # so that we select only the [CLS] token as sentence representation
    # just save it to pickles
    
    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    component_path = '../pickles/activated_components.pickle'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 20
    
    # Todo: generalize for every model 
    layers = [*range(0, 12, 1)]
    heads = [*range(0, 12, 1)]

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # test_mask()
    
    # model = RobertaModel.from_pretrained('../models/roberta.large.mnli', checkpoint_file='model.pt') 
    # bpe='../models/encoder.json')

    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased-mnli/")
    model = AutoModelForSequenceClassification.from_pretrained("../bert-base-uncased-mnli/")
    
    
    model = model.to(DEVICE)
    
    # model2 = AutoModelForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")
    
    # Todo: balance example between HOL and LOL by sampling from population
    experiment_set = ExperimentDataset(data_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                             encode = tokenizer
                            )

    labels = {"contradiction": 0 , "entailment" : 1, "neutral": 2}
    
    # model_name = '../models/roberta.large.mnli/model.pt'
    # model = torch.hub.load('pytorch/fairseq', model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Todo:  fixing hardcode of vocab.bpe and encoder.json for roberta fairseq
    
    # Todo: average score of each neuron's activation across batch
    
    dataloader = DataLoader(experiment_set, 
                            batch_size = 32,
                            shuffle = False, 
                            num_workers=0)
 
    # if not os.path.isfile(component_path):
    collect_output_components(model = model,
                            dataloader = dataloader,
                            tokenizer = tokenizer,
                            DEVICE = DEVICE,
                            layers = layers,
                            heads = heads)
    
    cma_analysis(path = component_path,
                model = model,
                layers = layers,
                heads  =  heads,
                tokenizer = tokenizer,
                experiment_set = experiment_set,
                DEVICE = DEVICE)
    
    # prunning(model = model,
    #          layers= [0, 1, 2, 3, 4])
    
    # Todo: collect output of every compontent in model 
    # 1. do-treatment : high overlap scores
    # 2. no-treament : low overlap scores

    
    # neuron_intervention(model = model,
    #                     tokenizer = tokenizer,
    #                     layers = [0, 1, 2, 3, 4],
    #                     neurons = [1,2,3],
    #                     dataloader = dataloader) 
                    
    

if __name__ == "__main__":
    main()

"""
X : population of text

note: we cannot random population because text already carry word overlap 
inside

treatment: high overlap score
no-treatment: low overlap score

X[do] : population treatment group
X[no] : population control group  

ATT:

eg. quantify the effect of Tylenol on headache status for people who 

treatment : Tylenol 
intermediate : taking the pill
effect : Headache 


Total effect:
 amount of bais under a gendered reading


In gender bias
    - captured specific model compontents on pred. eg. subspaces of contextual 
    word representations

u = The nurse said that {blank}; {} = he/she

base case: 

y_{null}(u) =  p(anti l- streotype | u) / p(streotype | u)

set u to an anti-streotype case:
    set-gender : nurse ->  man (anti-streriotype)
        need to  anti-steriotype compare ?

y_{set-gender}(u) =  p(anti-streotype | u) / p(streotype | u)


TE(set-gender, null;) = y_{set_gender} - y_{null} / y_{null}

note :

we cannot follow gender bias because the same unit cant perform counterfactual 
while our case counterfactual are missing so we need to compute in population level

"""
    


    
