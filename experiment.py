import os 
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import Intervention, get_overlap_thresholds, group_by_treatment


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


    def get_dataframe(self):

        return self.df


def main():


    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 50

    experiment_set = ExperimentDataset(data_path,
                             json_file,
                             upper_bound = upper_bound,
                             lower_bound = lower_bound,
                            )

    df = experiment_set.get_dataframe()


    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_name = 'bert-base-uncased'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    base_sentence = "The {} said that"
    biased_word = "teacher"
    
    
    
    breakpoint()

    # do-treatment, word overlap  more than 80 percent
    # no-treatment, word overlap less than 50 percent

    """
    intervention = Intervention(
            tokenizer,
            base_sentence,
            [biased_word, "man", "woman"],
            ["he", "she"],
            device=DEVICE)
    interventions = {biased_word: intervention}
    """

if __name__ == "__main__":
    main()