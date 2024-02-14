
import itertools
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from my_package.utils import Intervention, get_overlap_thresholds, group_by_treatment, test_mask, Classifier, get_hidden_representations
from my_package.utils import collect_output_components , report_gpu
from experiment import ExperimentDataset
from tqdm import tqdm
import os
import os.path
import pandas as pd
import random
import pickle
import json
import torch
import torch.nn as nn

layers = [11]

is_group_by_class =   True
is_averaged_embeddings =  False 
heads =  [*range(0, 12, 1)]
layers = [*range(0, 12, 1)]



valid_path = '../debias_fork_clean/debias_nlu_clean/data/nli/'
json_file = 'multinli_1.0_dev_matched.jsonl'

# used to compute nie scores
num_samples = 300 #3000

save_nie_set_path = f'../pickles/class_level_nie_{num_samples}_samples.pickle' if is_group_by_class else f'../pickles/nie_{num_samples}_samples.pickle'

# percent threshold of overlap score
upper_bound = 95
lower_bound = 5
    
label_maps = {"contradiction": 0 , "entailment" : 1, "neutral": 2}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased-mnli/")
model = AutoModelForSequenceClassification.from_pretrained("../bert-base-uncased-mnli/")
model = model.to(DEVICE)



  
counterfactual_representation_paths = []
is_counterfactual_exist = []

for component in tqdm(["Q","K","V","AO","I","O"], desc="Components"): 

    if is_averaged_embeddings:
            
        if is_group_by_class:

            cur_path = f'../pickles/avg_class_level_{component}_counterfactual_representation.pickle'

        else:
            cur_path = f'../pickles/avg_{component}_counterfactual_representation.pickle'

    else:

        if is_group_by_class:

            cur_path = f'../pickles/individual_class_level_{component}_counterfactual_representation.pickle'
        
        else:

            cur_path = f'../pickles/individual_{component}_counterfactual_representation.pickle'

    counterfactual_representation_paths.append(cur_path)
    is_counterfactual_exist.append(os.path.isfile(cur_path))


experiment_set = ExperimentDataset(valid_path,
                        json_file,
                        upper_bound = upper_bound,
                        lower_bound = lower_bound,
                        encode = tokenizer,
                        is_group_by_class = is_group_by_class,
                        num_samples = num_samples
                    )

dataloader = DataLoader(experiment_set, 
                batch_size = 32,
                shuffle = False, 
                num_workers=0)

# Todo: we want 4 pickles to save component I 
# component : HOL and LOL, 

# saving counterfactuals 
# counterfactual_representation_paths = collect_output_components(model = model,
#                                             counterfactual_paths = counterfactual_representation_paths,
#                                             experiment_set = experiment_set,
#                                             dataloader = dataloader,
#                                             tokenizer = tokenizer,
#                                             DEVICE = DEVICE,
#                                             layers = layers,
#                                             heads = heads,
#                                             is_averaged_embeddings = is_averaged_embeddings
#                                            )

hidden_representations = {}
        

for cur_path, component in zip(counterfactual_representation_paths, ["Q","K","V","AO","I","O"]):

    # if component not in ["I","O"]: continue

    if component not in hidden_representations.keys(): hidden_representations[component] = {}

    print(f"current component : {component}")

    if component == "I":

        for  do in ['High-overlap','Low-overlap']:
            
            if do not in hidden_representations[component].keys(): hidden_representations[component][do] = {}

            for class_name in ["contradiction","entailment","neutral"]:
            
                testing_path = f'../pickles/individual_{component}_{do}_{class_name}_counterfactual_representation.pickle'
                
                # hidden_representations[component][do][class_name][layer][sample_idx]
                with open(testing_path, 'rb') as handle:
                    hidden_representations[component][do][class_name] = pickle.load(handle)
                    print(f"loading from pickle {testing_path} !")        

                del hidden_representations[component][do][class_name]
                report_gpu()


    else:

        with open(cur_path, 'rb') as handle:
            hidden_representations[component] = pickle.load(handle)
            print(f"loading from pickle {cur_path} !")        

        
        del hidden_representations[component]
        report_gpu()


# # load counterfactual pickles 
# for component in ["I","O"]: 
#     #["Q","K","V","AO","I","O"]: 
    
#     counter = {}
#     NIE = {}

#     # print(f"============== start component :{component} : ===============")
#     print(f"before getting counterfactual :")
#     report_gpu()

#     counterfactual_components = get_hidden_representations(counterfactual_representation_paths, [11], heads, is_group_by_class, is_averaged_embeddings, component)
    
#     print(f"After getting counterfactual :")
#     report_gpu()