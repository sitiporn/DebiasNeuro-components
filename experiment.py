import os
import pandas as pd
import random
import pickle
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import Intervention, get_overlap_thresholds, group_by_treatment, neuron_intervention
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

def get_activation(layer, do, activation):
  
  # the hook signature
  def hook(model, input, output):
    
    if layer not in activation[do].keys():
        activation[do][layer] = []

    activation[do][layer].append(output.detach())
  
  return hook

def collect_output_components(model, dataloader, tokenizer, DEVICE):
   
    hooks =  {"do-treatment" : None, "no-treatment": None}
    
    # Todo: generalize for every model 
    layers = [*range(0, 12, 1)]
    heads = [*range(0, 12, 1)]
    
    
    # AO = lambda layer : model.bert.encoder.layer[layer].output.dense 
    # get attention weight output


    # self_attention = lambda layer : model.bert.encoder.layer[layer].attention.self
    self_output = lambda layer : model.bert.encoder.layer[layer].attention.output
    intermediate_layer = lambda layer : model.bert.encoder.layer[layer].intermediate
    output_layer = lambda layer : model.bert.encoder.layer[layer].output


    # using for register
    ao = {}
    intermediate = {}
    out = {} 

    #dicts to store the activations
    ao_activation = {}
    intermediate_activation = {}
    out_activation = {} 
    attention_data = {}        

    inputs = {}

    batch_idx = 0

    for pair_sentences , labels in tqdm(dataloader):

        if batch_idx == 4:
            #breakpoint()
            break
            

        for do in ['do-treatment','no-treatment']:

            if do not in ao_activation.keys():
                ao_activation[do] = {}
                intermediate_activation[do] = {}
                out_activation[do] = {} 
            
            pair_sentences[do] = [[premise, hypo] for premise, hypo in zip(pair_sentences[do][0], pair_sentences[do][1])]
            inputs[do] = tokenizer(pair_sentences[do], padding=True, truncation=True, return_tensors="pt")

            inputs[do] = {k: v.to(DEVICE) for k,v in inputs[do].items()}

            # breakpoint()
            
            # get attention weight 
            outputs = model(**inputs[do], output_attentions = True)

            logits = outputs.logits
            # labels 0: contradiction, 1: entailment, 2: neutral
            # predictions = logits.argmax(dim=1)
            # print(f"current prediction of {do} : {predictions[:10]}")
            # print(f"current labels {do} : {labels[do][:10]}")

            if do not in attention_data.keys():
                attention_data[do] = [outputs.attentions]
            else:
                attention_data[do].append(outputs.attentions)

            # register forward hooks on all layers
            for layer in layers:

                ao[layer] = self_output(layer).register_forward_hook(get_activation(layer, do, ao_activation))
                intermediate[layer] = intermediate_layer(layer).register_forward_hook(get_activation(layer, do, intermediate_activation))
                out[layer] = output_layer(layer).register_forward_hook(get_activation(layer, do, out_activation))

            # get activatation
            outputs = model(**inputs[do])
 
            # detach the hooks
            for layer in layers:
                ao[layer].remove()
                intermediate[layer].remove()
                out[layer].remove()

        
        batch_idx += 1



    with open('../pickles/activated_components.pickle', 'wb') as handle:

        pickle.dump(ao_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(intermediate_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(out_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(attention_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"save activate components done ! ")
    
    
    # note: cannot concate because batching different seq_len
    # just save it to pickles

        
        #attention_override = model(batch, target_mapping=target_mapping)[-1]

    # for do in output.keys():
    #     hooks[do] = {}
    #     # Todo: batching  for dataloader
    #     for layer in layers:
    #         # get output of all attention heads and FFN 
    #         hooks[do][layer]  = {}
            # for module in ["attentions","l_AO","l_intermediates","l_outputs"]:
            # with torch.no_grad():
                #hooks[do][layer]["attentions"] = []
                
                # hooks[do][layer]["attentions"].append(self_attention(layer).register_forward_hook(
                #                             attention_intervention(override_attention, attn_override_mask)))

                # for hook in hooks: hook.remove()

                # ,"l_AO","l_intermediates","l_outputs"]:


def main():

    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 20

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model = RobertaModel.from_pretrained('../models/roberta.large.mnli', checkpoint_file='model.pt') 
    # bpe='../models/encoder.json')
    
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased-mnli/")
    model = AutoModelForSequenceClassification.from_pretrained("../bert-base-uncased-mnli/")
    model = model.to(DEVICE)

    # model2 = AutoModelForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")

    
    #
    
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
 
    # collect_output_components(model = model,
    #                          dataloader = dataloader,
    #                          tokenizer = tokenizer,
    #                          DEVICE = DEVICE)


    with open('../pickles/activated_components.pickle', 'rb') as handle:
        ao_activation = pickle.load(handle)
        intermediate_activation = pickle.load(handle)
        out_activation = pickle.load(handle)
        attention_data = pickle.load(handle)

    breakpoint()



    
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
    


    
