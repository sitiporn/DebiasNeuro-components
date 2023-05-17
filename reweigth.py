import os
import os.path
import pandas as pd

dev_path = "../debias_fork_clean/debias_nlu_clean/data/nli/"
file = 'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl'

result_path = os.path.join(os.path.join(dev_path, file))

df = pd.read_json(result_path, lines=True)

breakpoint()


"""
this dev file that we use 
dev_json = new_dev_df.to_json(orient='records', lines=True)    

with open('dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl', 'w') as json_file:
    json_file.write(dev_json)
dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl
---
using this as metric to quantifying by

1. main model correct, bias model correct -> low loss
2. main model 

bias model -> correct   -> low weight
bias model -> incorrect -> high weight 


reweight probs ;  only the class belonging to the golden answers

"""
