import sys
import pickle
from tqdm import tqdm
import os 
import yaml
import torch
import numpy as np
from utils import get_ans
from data import convert_to_text_ans

def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"

def get_result(config, epsilons, eval_path, prediction_path, neuron_path, top_neuron, digits, prediction_mode):
    
    if config['to_text']: convert_to_text_ans(config, neuron_path)

    num_neuron_groups = [config['neuron_group']] if config['neuron_group'] is not None else list(top_neuron.keys())

    for epsilon in (t := tqdm(epsilons)):  

        epsilon_path = f'v{round(epsilon, digits)}'

        t.set_description(f"epsilon : {epsilon} ")
        
        for group in num_neuron_groups:

            text_answer_path = f'txt_answer_{prediction_mode}_{config["eval"]["intervention_mode"]}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
            result_path = f'result_{prediction_mode}_{config["eval"]["intervention_mode"]}_L{config["layer"]}_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

            if config['eval']['all_layers']: text_answer_path = f'txt_answer_{prediction_mode}_{config["eval"]["intervention_mode"]}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  
            if config['eval']['all_layers']: result_path = f'result_{prediction_mode}_{config["eval"]["intervention_mode"]}_all_layers_{group}-k_{config["eval"]["do"]}_{config["intervention_type"]}_{config["dev-name"]}.txt'  

            config['evaluations'][group] = {}

            text_answer_path = os.path.join(os.path.join(prediction_path, epsilon_path),  text_answer_path)
            result_path = os.path.join(os.path.join(eval_path, epsilon_path),  result_path)

            tables = {}

            fi = open(text_answer_path, "r")

            first = True
            guess_dict = {}

            for line in fi:
                if first:
                    first = False
                    continue
                else:
                    parts = line.strip().split(",")
                    guess_dict[parts[0]] = format_label(parts[1])

            # load from hans set up
            fi = open("../hans/heuristics_evaluation_set.txt", "r")

            correct_dict = {}
            first = True

            heuristic_list = []
            subcase_list = []
            template_list = []

            for line in fi:
                if first:
                    labels = line.strip().split("\t")
                    idIndex = labels.index("pairID")
                    first = False
                    continue
                else:
                    parts = line.strip().split("\t")
                    this_line_dict = {}
                    for index, label in enumerate(labels):
                        if label == "pairID":
                            continue
                        else:
                            this_line_dict[label] = parts[index]
                    correct_dict[parts[idIndex]] = this_line_dict

                    if this_line_dict["heuristic"] not in heuristic_list:
                        heuristic_list.append(this_line_dict["heuristic"])
                    if this_line_dict["subcase"] not in subcase_list:
                        subcase_list.append(this_line_dict["subcase"])
                    if this_line_dict["template"] not in template_list:
                        template_list.append(this_line_dict["template"])

            heuristic_ent_correct_count_dict = {}
            subcase_correct_count_dict = {}
            template_correct_count_dict = {}
            heuristic_ent_incorrect_count_dict = {}
            subcase_incorrect_count_dict = {}
            template_incorrect_count_dict = {}
            heuristic_nonent_correct_count_dict = {}
            heuristic_nonent_incorrect_count_dict = {}



            for heuristic in heuristic_list:
                heuristic_ent_correct_count_dict[heuristic] = 0
                heuristic_ent_incorrect_count_dict[heuristic] = 0
                heuristic_nonent_correct_count_dict[heuristic] = 0 
                heuristic_nonent_incorrect_count_dict[heuristic] = 0

            for subcase in subcase_list:
                subcase_correct_count_dict[subcase] = 0
                subcase_incorrect_count_dict[subcase] = 0

            for template in template_list:
                template_correct_count_dict[template] = 0
                template_incorrect_count_dict[template] = 0

            for key in correct_dict:
                traits = correct_dict[key]
                heur = traits["heuristic"]
                subcase = traits["subcase"]
                template = traits["template"]

                guess = guess_dict[key]
                correct = traits["gold_label"]

                if guess == correct:
                    if correct == "entailment":
                        heuristic_ent_correct_count_dict[heur] += 1
                    else:
                        heuristic_nonent_correct_count_dict[heur] += 1

                    subcase_correct_count_dict[subcase] += 1
                    template_correct_count_dict[template] += 1
                else:
                    if correct == "entailment":
                        heuristic_ent_incorrect_count_dict[heur] += 1
                    else:
                        heuristic_nonent_incorrect_count_dict[heur] += 1
                    subcase_incorrect_count_dict[subcase] += 1
                    template_incorrect_count_dict[template] += 1

            tables['correct']  = { 'entailed': heuristic_ent_correct_count_dict, 'non-entailed': heuristic_nonent_correct_count_dict}
            tables['incorrect'] = { 'entailed': heuristic_ent_incorrect_count_dict,  'non-entailed': heuristic_nonent_incorrect_count_dict}

            for cur_class in ['entailed','non-entailed']:

                print(f"Heuristic  {cur_class} results:")

                if cur_class not in config["evaluations"][group].keys():  config["evaluations"][group][cur_class] = {}

                for heuristic in heuristic_list:

                    correct = tables['correct'][cur_class][heuristic]
                    incorrect = tables['incorrect'][cur_class][heuristic]

                    total = correct + incorrect
                    percent = correct * 1.0 / total
                    print(heuristic + ": " + str(percent))

                    config["evaluations"][group][cur_class][heuristic] = percent

            
            with open(result_path, 'wb') as handle: 
                pickle.dump(config["evaluations"], handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'saving evaluation predictoins into : {result_path}')



# ++++++  config ++++++++++++

with open("config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(config)

low  =  config['params']['low'] #.7945 0.785  0.75   
high =  config['params']['high']  #.7955 0.795  0.85
step =  config['params']['step'] 
digits = len(str(step).split('.')[-1])
size= config['params']['size']

# use to read prediction txt files
top_mode =  'percent' if config['k'] is not None  else 'neurons'
prediction_mode = 'percent' if config['k'] is not None  else config['weaken'] if config['weaken'] is not None else 'neurons'

eval_path = f'../pickles/evaluations/'
prediction_path = '../pickles/prediction/' 

neuron_path = f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_all_layers.pickle' if config['eval']['all_layers'] else f'../pickles/top_neurons/top_neuron_{top_mode}_{config["eval"]["do"]}_{config["layer"]}.pickle'

if config['intervention_type'] == "remove": epsilons = (low - high) * torch.rand(size) + high  # the interval (low, high)
if config['intervention_type'] == "weaken": epsilons = [config['weaken']] if config['weaken'] is not None else [round(val, digits)for val in np.arange(low, high, step).tolist()]
if config['intervention_type'] not in ["remove","weaken"]: epsilons = [0]

with open(neuron_path, 'rb') as handle: 
    top_neuron = pickle.load(handle)

if config['get_result']: get_result(config, eval_path, prediction_path, neuron_path, top_neuron, digits, prediction_mode)


            