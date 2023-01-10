import os 
import pandas as pd
import json
import torch
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel
from torch.utils.data import Dataset, DataLoader
from utils import Intervention, get_overlap_thresholds, group_by_treatment

"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
"""

class ExperimentDataset(Dataset):
    def __init__(self, data_path, json_file,upper_bound, lower_bound, encode ,DEBUG = False) -> None:
        
        data_path = os.path.join(data_path,json_file)

        self.df = pd.read_json(data_path, lines=True) 

        self.encode = encode

        self.premises = {"do-treatment": None, 
                         "no-treatment": None,
                        }

        self.hypothesises = {"do-treatment": None, 
                             "no-treatment" : None
                             }
        exp_set = {"do-treatment": None,
                   "no-treatment": None}

        self.labels =   {"do-treatment": None,
                        "no-treatment": None
                        }

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
        return len(self.intervention.batch_tok)
    
    def __getitem__(self, idx):

        sentence_pair = self.intervention.batch_tok[idx]
        label = self.labels['do-treatment'][idx]
        
        return sentence_pair, label


def neuron_intervention(model,
                        layers,
                        neurons,
                        dataloader,
                        intervention_type='replace'):

        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output):

            # overwrite value in the output
            # define mask to overwrite
            scatter_mask = torch.zeros_like(output, dtype = torch.bool)
            
            print(f"set of neurons to intervene {neurons}")
            
            # where to intervene
            scatter_mask[:,:, neurons] = 1

            # value to replace : (seq_len, batch_size, output_dim)
            value = torch.zeros_like(output, dtype = torch.float)

            # (bz, seq_len, input_dim) @ (input_dim, output_dim)
            #  seq_len, batch_size, hidden_dim 
            
            """
            print(f"== inside intervention hook ==")
            print(f"output shape : {output.shape} "ji)
            print(f"scatter_mask : {scatter_mask.shape}")
            output.masked_scatter_(scatter_mask, value)
            
            """
            
        neuron_layer = lambda layer : model.model.encoder.sentence_encoder.layers[layer].final_layer_norm
        
        handle_list = []

        for batch_idx, (sentence_pairs, label) in enumerate(dataloader):

            print(f"batch_idx : {batch_idx}")
            print(f"current sentence pair : {sentence_pairs}")
            print(f"current label : {label}")

            #for layer in layers:
                #handle_list.append(neuron_layer(layer).register_forward_hook(intervention_hook))

            #new_logprobs = model.predict('mnli', intervention.batch_tok[0:8])
            #predictions = new_logprobs.argmax(dim=1)
        
            #print(f"=== with intervene ====")
            #print(new_logprobs[:8,:])
            #print(predictions)

            #for hndle in handle_list:
                #hndle.remove() 
            
            #logprobs = model.predict('mnli', intervention.batch_tok[0:8])
            #predictions = logprobs.argmax(dim=1)
            
            #print(f"=== without intervene ====")
            #print(logprobs[:8,:])
            #print(predictions)



def main():

    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 20

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = RobertaModel.from_pretrained('../models/roberta.large.mnli', checkpoint_file='model.pt') #, bpe='../models/encoder.json')

    model = model.to(DEVICE)

    experiment_set = ExperimentDataset(data_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                             encode = model.encode
                            )

    
    # model_name = '../models/roberta.large.mnli/model.pt'
    # model = torch.hub.load('pytorch/fairseq', model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Todo:  fixing hardcode of vocab.bpe and encoder.json

    """
    Research Question: intervene entire seq of text is the right thing ? 
        1. entire seq -> the representation ?
        2. intervene only hypothesises word constructed from a premise ? 

    shortcut: 
        - assume that a premise entail hypothesises constructed from words in the premise
    
    neuron output : [seq_len, batch_size, out_dim] 
    
    """
    
    # Todo: average score of each neuron's activation across batch
    
    dataloader = DataLoader(experiment_set, batch_size=4,
                        shuffle = False, num_workers=0)

    neuron_intervention(model = model,
                        layers = [9, 8, 12 ,16, 18, 20],
                        neurons = [1,2,3],
                        dataloader = dataloader) 

if __name__ == "__main__":
    main()
    
    
    


"""
implemtent fewshot havnt done yet

"""
    