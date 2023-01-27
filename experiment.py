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
from utils import collect_output_components, report_gpu
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F

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
        
        # get [CLS] activation 
        q_cls = pickle.load(handle)
        k_cls = pickle.load(handle)
        v_cls = pickle.load(handle)
        ao_cls = pickle.load(handle)
        intermediate_cls = pickle.load(handle)
        out_cls = pickle.load(handle)
        
        attention_data = pickle.load(handle)
        counter = pickle.load(handle)

    # get average of [CLS] activations
    q_cls_avg = {}
    k_cls_avg = {}
    v_cls_avg = {}
    ao_cls_avg = {}
    intermediate_cls_avg = {}
    out_cls_avg = {}
    
    for do in ['do-treatment','no-treatment']:

        q_cls_avg[do] = {}
        k_cls_avg[do] = {}
        v_cls_avg[do] = {}
        ao_cls_avg[do] = {}
        intermediate_cls_avg[do] = {}
        out_cls_avg[do] = {}
         
         # concate all batches
        for layer in layers:

            # compute average over samples
            q_cls_avg[do][layer] = q_cls[do][layer] / counter
            k_cls_avg[do][layer] = k_cls[do][layer] / counter
            v_cls_avg[do][layer] = v_cls[do][layer] / counter

            ao_cls_avg[do][layer] = ao_cls[do][layer] / counter
            intermediate_cls_avg[do][layer] = intermediate_cls[do][layer] / counter
            out_cls_avg[do][layer] = out_cls[do][layer] / counter

    return q_cls_avg, k_cls_avg, v_cls_avg, ao_cls_avg, intermediate_cls_avg,  out_cls_avg

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

    pairs = {}

    # For every samples
    # E(Yi|Xi=1)−E(Yi|Xi=0) 

    # guess prediction
    # for pair_sentences , labels in tqdm(dataloader):

    # Todo get pair sentences of all overlap scores 

    pairs["entailment"] = list(experiment_set.df[experiment_set.df.gold_label == "entailment"].pair_label)
    
    #pair_labels = list(experiment_set.df.pair_label)

    dataset = [([premise, hypo], label) for idx, (premise, hypo, label) in enumerate(pairs['entailment'])]
    dataloader = DataLoader(dataset, batch_size=32)

    q_layer = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    k_layer = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    v_layer = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    
    # self_attention = lambda layer : model.bert.encoder.layer[layer].attention.self
    self_output = lambda layer : model.bert.encoder.layer[layer].attention.output
    intermediate_layer = lambda layer : model.bert.encoder.layer[layer].intermediate
    output_layer = lambda layer : model.bert.encoder.layer[layer].output

    for i, (sentences, labels) in enumerate(dataloader):

        premise, hypo = sentences

        pair_sentences = [[premise, hypo] for premise, hypo in zip(sentences[0], sentences[1])]

        distributions = {}
        
        inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}

        outputs = model(**inputs)
        distributions['normal'] = F.softmax(outputs.logits ,dim=-1)
        distributions['intervene'] = {}

        print(f"normal prediction : ")
        print(distributions['normal'])

        # run one full neuron intervention experiment
        for do in ['do-treatment','no-treatment']:
            # Todo : add qeury, key and value 
            for layer in layers:
                
                distributions['intervene'][layer] = {}
                
                # get each prediction for nueron intervention
                for neuron_id in range(ao_cls_avg[do][layer].shape[0]):
                    
                    # select layer to register  and input which neurons to intervene
                    hooks = [] 

                    neuron_ids = [*range(0, ao_cls_avg[do][layer].shape[0], 1)]

                    # dont forget to change 11th to layer variable !
                    hooks.append(self_output(layer).register_forward_hook(neuron_intervention(neuron_ids = neuron_ids, 
                                                                              value = ao_cls_avg[do][11])))
                
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    distributions['intervene'][layer][neuron_id] = F.softmax(outputs.logits ,dim=-1)

                    # print(f"intervene prediction")
                    # print(predictions.eq(new_predictions))
                    
                    for hook in hooks: hook.remove() 


                for nueron_id in range(intermediate_cls_avg[do][layer].shape[0]):

                    hooks = [] 
                    hooks.append(intermediate_layer(layer).register_forward_hook(neuron_intervention()))

                    outputs = model(**inputs)
                    logits = outputs.logits
                    predictions = logits.argmax(dim=1)
                    
                    for hook in hooks: hook.remove() 
                
                for nueron_id in range(out_cls_avg[do][layer].shape[0]):
                    
                    hooks = [] 

                    neuron_ids = [*range(0, out_cls_avg[do][layer].shape[0], 1)]
                    
                    # dont forget to change 11th to layer variable !
                    hooks.append(output_layer(layer).register_forward_hook(neuron_intervention(neuron_ids = neuron_ids, 
                                                                    value = out_cls_avg[do][11])))
                        
                    outputs = model(**inputs)
                    logits = outputs.logits
                    new_predictions = logits.argmax(dim=1)

                    # print(f"intervene prediction")
                    # print(new_predictions)

                    # print(f"== compare result ==")
                    # print(predictions.eq(new_predictions))
                    # print(f"# not change : {torch.sum(predictions.eq(new_predictions))}")
                    
                    for hook in hooks: hook.remove() 

                    breakpoint()

def main():

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

    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased-mnli/")
    model = AutoModelForSequenceClassification.from_pretrained("../bert-base-uncased-mnli/")
    model = model.to(DEVICE)
    
    # Todo: balance example between HOL and LOL by sampling from population
    experiment_set = ExperimentDataset(data_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                             encode = tokenizer
                            )

    labels = {"contradiction": 0 , "entailment" : 1, "neutral": 2}
    
    # Todo:  fixing hardcode of vocab.bpe and encoder.json for roberta fairseq
    
    dataloader = DataLoader(experiment_set, 
                            batch_size = 32,
                            shuffle = False, 
                            num_workers=0)

 
    if not os.path.isfile(component_path):
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

