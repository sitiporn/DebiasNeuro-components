import os 
import pandas as pd
import json
from utils import get_overlap_score

data_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
json_file = 'multinli_1.0_train.jsonl'

data_path = os.path.join(data_path,json_file)

df = pd.read_json(data_path, lines=True) 

print(df.columns)

pair_and_label = []

for i in range(len(df)):
    pair_and_label.append((df['sentence1'][i], df['sentence2'][i], df['gold_label'][i]))

df['pair_label'] = pair_and_label

df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)
