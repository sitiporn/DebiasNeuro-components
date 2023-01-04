import os 
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import Intervention, get_overlap_thresholds, group_by_treatment


def main():


    DEBUG = True
    
    data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
    json_file = 'multinli_1.0_train.jsonl'
    upper_bound = 80
    lower_bound = 50


    data_path = os.path.join(data_path,json_file)

    df = pd.read_json(data_path, lines=True) 

    if DEBUG: print(df.columns)

    pair_and_label = []
    for i in range(len(df)):
        pair_and_label.append((df['sentence1'][i], df['sentence2'][i], df['gold_label'][i]))

    df['pair_label'] = pair_and_label
    thresholds = get_overlap_thresholds(df, upper_bound, lower_bound)
    df['is_treatment'] = df.apply(lambda row: group_by_treatment(thresholds, row.overlap_scores, row.gold_label), axis=1)


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_name = 'bert-base-uncased'
    
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    base_sentence = "The {} said that"
    biased_word = "teacher"

    breakpoint()

    # Todo : load entailment dataset 
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