import os 
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
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
    def __init__(self, data_path, json_file,upper_bound, lower_bound, encode ,DEBUG = False) -> None:
        
        data_path = os.path.join(data_path,json_file)

        self.df = pd.read_json(data_path, lines=True) 

        self.encode = encode

        self.premises = {}
        self.hypothesises = {}
        self.labels =   {}
        pair_and_label = []
        
        if DEBUG: print(self.df.columns)

        for i in range(len(self.df)):
            pair_and_label.append((self.df['sentence1'][i], self.df['sentence2'][i], self.df['gold_label'][i]))

        
        self.df['pair_label'] = pair_and_label
        thresholds = get_overlap_thresholds(self.df, upper_bound, lower_bound)

        # get w/o treatment on entailment set
        self.df['is_treatment'] = self.df.apply(lambda row: group_by_treatment(thresholds, row.overlap_scores, row.gold_label), axis=1)

        self.exp_set = {"do-treatment": self.get_intervention_set(),
                        "no-treatment": self.get_base_set()}

        

        for op in ["do-treatment","no-treatment"]:
            
            self.premises[op] = list(self.exp_set[op].sentence1)
            self.hypothesises[op] = list(self.exp_set[op].sentence2)
            self.labels[op] = list(self.exp_set[op].gold_label)

        ## word overlap  more than 80 percent
        #self.premises["do-treatment"] = list(intervention_set.sentence1)
        #self.hypothesises["do-treatment"] = list(intervention_set.sentence2)

        ## word overlap less than 20 percent
        #self.premises["no-treatment"] = list(base_set.sentence1)
        #self.hypothesises["no-treatment"] = list(base_set.sentence2)
    
        self.intervention  = Intervention(encode = self.encode,
                                 premises = self.premises["do-treatment"],
                                 hypothesises = self.hypothesises["do-treatment"] 
                                 )


    def get_intervention_set(self):

        # get high overlap score pairs
        return self.df[self.df['is_treatment'] == "treatment"]

    def get_base_set(self):

        # get low overlap score pairs
        return self.df[self.df['is_treatment'] == "no-treatment"]

    def __len__(self):
        # Todo: generalize label
        return len(self.intervention.pair_sentences)
    
    def __getitem__(self, idx):

        pair_sentences = self.intervention.pair_sentences[idx]
        label = self.labels['do-treatment'][idx]
        
        return pair_sentences , label


def prunning(model, layers):

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


        # Todo: set which component to intervene
        with torch.no_grad(): 
            model.bert.encoder.layer[layer].attention.self.query.weight.data.copy_(Wl_Q(layer) *  Ml_Q )
            # model.bert.encoder.layer[layer].attention.self.key.weight = Wl_K(layer) *  Ml_K 
            # model.bert.encoder.layer[layer].attention.self.key.value = Wl_V(layer) *  Ml_V 
            # model.bert.encoder.layer[layer].output.dense.weight = Wl_AO(layer) *  Ml_AO
            # model.bert.encoder.layer[layer].intermediate.dense = Wl_I(layer) *  Ml_I 
            # model.bert.encoder.layer[layer].output.dense = Wl_O(layer) *  Ml_O 


        breakpoint()



def main():

    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 20

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model = RobertaModel.from_pretrained('../models/roberta.large.mnli', checkpoint_file='model.pt') 
    # bpe='../models/encoder.json')
    
    tokenizer = AutoTokenizer.from_pretrained("ishan/bert-base-uncased-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("ishan/bert-base-uncased-mnli")
    # model = model.to(DEVICE)
    
    # Todo: balance example between HOL and LOL
    experiment_set = ExperimentDataset(data_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                             encode = tokenizer
                            )

    df_entail =  experiment_set.df[experiment_set.df['gold_label'] == "entailment"]
    df_contradiction = experiment_set.df[experiment_set.df['gold_label'] == "contradiction"]
    df_neutral = experiment_set.df[experiment_set.df['gold_label'] == "neutral"]

    labels = {"contradiction": 0 , "entailment" : 1, "neutral": 2}
    
    
    # model_name = '../models/roberta.large.mnli/model.pt'
    # model = torch.hub.load('pytorch/fairseq', model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Todo:  fixing hardcode of vocab.bpe and encoder.json for roberta fairseq
    
    # Todo: average score of each neuron's activation across batch
    
    dataloader = DataLoader(experiment_set, batch_size=4,
                        shuffle = False, num_workers=0)
 
    prunning(model = model,
             layers= [0, 1, 2, 3, 4])
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
    


    
