import os 
import pandas as pd
import json
from utils import get_overlap_score
import numpy as np

def get_overlap_thresholds(df, upper_bound, lower_bound):
    
    thresholds = {"treatment": None, "no-treatment": None}

    df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)

    # Todo: get overlap_score for whole entailment sets
    entail_mask = (df['gold_label'] == "entailment").tolist()
    overlap_scores = df['overlap_scores'][entail_mask]

    thresholds["no-treatment"] = np.percentile(overlap_scores, lower_bound)
    thresholds["treatment"] = np.percentile(overlap_scores, upper_bound)

    return thresholds
    
def group_by_treatment(thresholds, overlap_score, gold_label):
    
    if gold_label == "entailment":
        if overlap_score >= thresholds['treatment']:
            return "treatment"
        elif overlap_score <= thresholds['no-treatment']:              
            return "no-treatment"
        else:
            return "exclude"
    else:
        return "exclude"


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

breakpoint()


