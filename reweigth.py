import os
import os.path
import pandas as pd

# 3 class {entailment: 0, contrandiction: 1, neatral: 2}
def relabel(label):

    if label == 'contradiction':
        return 1
    elif label == 'neutral':
        return 2
    else:
        return 0

# ps
def give_weight(label, probs): 

    golden_id = relabel(label)

    probs = probs[golden_id]

    return 1 / probs


dev_path = "../debias_fork_clean/debias_nlu_clean/data/nli/"

file = 'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl'

result_path = os.path.join(os.path.join(dev_path, file))

#Index(['gold_label', 'sentence1', 'sentence2', 'bias_probs'], dtype='object')
df = pd.read_json(result_path, lines=True)


# df['prob_score'] = df.apply(lambda x: [1, 2], axis=1)
df['weight_score'] = df[['gold_label', 'bias_probs']].apply(lambda x: give_weight(*x), axis=1)

avg_losses = []

# quantifying by using losses
for index, row in df.iterrows():

    gold_label = row['gold_label'] 
    bias_prob = row['bias_probs']
    weight_score = row['weight_score']

    print(f"gold_label : {gold_label}, bias_prob : {bias_prob}, weight score : {weight_score}")
    
    # using total loss to quantifying the impact 
    # 1. reweight probs from bias model
    # 2. aggregate with total loss
    # 3. using loss to do hyper parameter search for weaken rate with fixed masking rates
    # where is loss ? 

    # breakpoint()


# Todo: get label maps 
# check from prediction whether correct or not 
# incorrect; ex 1/0.3 
#   low prob -> high weight 
# correct 1/0.999 -> 1
#   high prob -> low weight    


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
