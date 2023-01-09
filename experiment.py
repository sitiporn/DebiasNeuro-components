import os 
import pandas as pd
import json
import torch
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel
from torch.utils.data import Dataset
from utils import Intervention, get_overlap_thresholds, group_by_treatment

"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
"""

class ExperimentDataset(Dataset):
    def __init__(self, data_path, json_file,upper_bound, lower_bound, DEBUG = False) -> None:
        
        data_path = os.path.join(data_path,json_file)

        self.df = pd.read_json(data_path, lines=True) 

        if DEBUG: print(self.df.columns)

        pair_and_label = []

        for i in range(len(self.df)):
            pair_and_label.append((self.df['sentence1'][i], self.df['sentence2'][i], self.df['gold_label'][i]))

        self.df['pair_label'] = pair_and_label
        thresholds = get_overlap_thresholds(self.df, upper_bound, lower_bound)

        # get w/o treatment on entailment set
        self.df['is_treatment'] = self.df.apply(lambda row: group_by_treatment(thresholds, row.overlap_scores, row.gold_label), axis=1)

    def get_intervention_set(self):

        # get high overlap score pairs
        return self.df[self.df['is_treatment'] == "treatment"]

    def get_base_set(self):

        # get low overlap score pairs
        return self.df[self.df['is_treatment'] == "no-treatment"]

def neuron_intervention(model,
                        layers,
                        neurons,
                        intervention,
                        intervention_type='replace'):
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output):

            # overwrite value in the output
            # define mask where to overwrite
            scatter_mask = torch.zeros_like(output, dtype = torch.bool)

            # value to replace : (seq_len, batch_size, output_dim)
            value = torch.zeros_like(output, dtype = torch.float)

            print("output before intervene")
            print(output[:3,:3, 995:1005])

            scatter_mask[:,:,:] = 1

            # (bz, seq_len, input_dim) @ (input_dim, output_dim)
            #  seq_len, batch_size, hidden_dim 
            print(f"== inside intervention hook ==")
            print(f"output shape : {output.shape} ")
            print(f"scatter_mask : {scatter_mask.shape}")
            output.masked_scatter_(scatter_mask, value)

            print("output after intervene")
            print(output[:3,:3, 995:1005])

            
        neuron_layer = lambda layer : model.model.encoder.sentence_encoder.layers[layer].final_layer_norm
        
        handle_list = []

        for layer in layers:
            handle_list.append(neuron_layer(layer).register_forward_hook(intervention_hook))
            break

        new_logprobs = model.predict('mnli',intervention.batch_tok[0:8])
        predictions = new_logprobs.argmax(dim=1)
        print(f"=== with intervene ====")
        print(new_logprobs[:3,:])
        print(predictions)

        for hndle in handle_list:
            hndle.remove() 
        
        logprobs = model.predict('mnli',intervention.batch_tok[0:8])
        predictions = logprobs.argmax(dim=1)
        
        print(f"=== without intervene ====")
        print(logprobs[:3,:])
        print(predictions)

        breakpoint()


def main():

    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 20

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    premises = {"do-treatment": None, 
                "no-treatment": None}

    hypothesises = {"do-treatment": None, 
                    "no-treatment" : None}

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    experiment_set = ExperimentDataset(data_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                            )

    intervention_set = experiment_set.get_intervention_set()
    base_set = experiment_set.get_base_set()

    # word overlap  more than 80 percent
    premises["do-treatment"] = list(intervention_set.sentence1)
    hypothesises["do-treatment"] = list(intervention_set.sentence2)


    # word overlap less than 20 percent
    premises["no-treatment"] = list(base_set.sentence1)
    hypothesises["no-treatment"] = list(base_set.sentence2)
    
    # model_name = '../models/roberta.large.mnli/model.pt'
    # model = torch.hub.load('pytorch/fairseq', model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Todo:  fixing hardcode of vocab.bpe and encoder.json
    model = RobertaModel.from_pretrained('../models/roberta.large.mnli', checkpoint_file='model.pt') #, bpe='../models/encoder.json')

    model = model.to(DEVICE)


    intervention  = Intervention(encode = model.encode,
                                 premises = premises["do-treatment"],
                                 hypothesises = hypothesises["do-treatment"] 
                                )
    
    """
    note: do we need to intervene entire sequence of text 

    neuron output : [seq_len, batch_size, out_dim] 
    
    """
    # Todo: average score of each neuron's activation across batch
    neuron_intervention(model = model,
                        layers = [23],
                        neurons = [1,2,3],
                        intervention = intervention)
    

if __name__ == "__main__":
    main()