import sys
import pickle
from tqdm import tqdm
import os 

def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"
# ++++++  config ++++++++++++
do = 'High-overlap'
layer = 1
debug = False
intervention_type = "weaken"
all_layers = True
dev_set = 'hans'

# use to read prediction txt files
prediction_path = '../pickles/prediction/' 
evaluations  = {}
intervene_mode = "Intervene"
k = None
best_weaken_val = 0.7936
best_neuron_group = 45
top_mode =  'percent' if k is not None  else 'neurons'
prediction_mode = 'percent' if k is not None  else best_weaken_val if best_weaken_val is not None else 'neurons'

eval_path = f'../pickles/evaluations/topk_{do}_all_layers_{intervention_type}.pickle'if all_layers else f'../pickles/evaluations/topk_{do}_L{layer}_{intervention_type}.pickle'
neuron_path = f'../pickles/top_neurons/top_neuron_{top_mode}_{do}_all_layers.pickle' if all_layers else f'../pickles/top_neurons/top_neuron_{top_mode}_{do}_{layer}.pickle'

with open(neuron_path, 'rb') as handle: top_neuron = pickle.load(handle)
num_neuron_groups = [best_neuron_group] if best_neuron_group is not None else list(top_neuron.keys())


# ++++++++++++++++++++++++++++++++

for group in (t := tqdm(num_neuron_groups)):
    
    text_answer_path = f'txt_answer_{prediction_mode}_{intervene_mode}_L{layer}_{group}-k_{do}_{intervention_type}_{dev_set}.txt'  

    if all_layers: text_answer_path = f'txt_answer_{prediction_mode}_{intervene_mode}_all_layers_{group}-k_{do}_{intervention_type}_{dev_set}.txt'  

    evaluations[group] = {}
    
    tables = {}
    
    t.set_description(f": Top {text_answer_path}")
    
    text_answer_path  = os.path.join(prediction_path,  text_answer_path)

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

        if cur_class not in evaluations[group].keys():  evaluations[group][cur_class] = {}
        
        for heuristic in heuristic_list:
            
            correct = tables['correct'][cur_class][heuristic]
            incorrect = tables['incorrect'][cur_class][heuristic]
            
            total = correct + incorrect
            percent = correct * 1.0 / total
            print(heuristic + ": " + str(percent))

            evaluations[group][cur_class][heuristic] = percent

with open(eval_path, 'wb') as handle: 
    pickle.dump(evaluations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'saving evaluation predictoins into : {eval_path}')


