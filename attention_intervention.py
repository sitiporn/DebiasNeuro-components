from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
from utils import BertAttentionOverride
import torch.nn.functional as F


# Step of previous project
# 1. training main models: using bias model's  predicted scores to aggregrate in training main models for every methods
#    - objective : let's model learn from low bias samples

# 2. inference: every models do counterfactual

# ** bias models used in training step and inference are the same

"""
Todo: 
 1. previous proj: get qqp file jsonl
    - Create features for training the bias model
    - train bias model used in the paper
 2. current proj : using model from previous project and performing intervene on model and get inference result

"""

# information structure
#  model name : normal prediction, intervene model's predictions


def attention_intervention(attn_override, attn_override_mask):


    def intervention_hook(module, input, outputs):

        attention_override_module = BertAttentionOverride(module, attn_override, attn_override_mask)

        print(f"-------------------------------")
        print(*input)
        print(f"-------------------------------")

        hidden_states = input[0] 
        attention_mask = input[1]
        
        return attention_override_module(hidden_states, attention_mask)

    return intervention_hook

model_name = 'bert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, output_hidden_states = True, output_attentions = True)

shape = (1, 12, 6, 6)
intervene_layer = 10

# attn_override_mask = torch.ones(shape, dtype=torch.bool)
# override_attention = torch.zeros(shape) 

attn_override_mask = torch.ones(outputs.attentions[intervene_layer][:,:,:6,:6].shape, dtype=torch.bool)
override_attention =  outputs.attentions[intervene_layer][:,:,:6,:6] 

# hidden_state = outputs.hidden_states

# Todo: create forward hook
breakpoint()


with torch.no_grad():
    
    hooks = []
    hooks.append(model.bert.encoder.layer[intervene_layer].attention.self.register_forward_hook(attention_intervention(override_attention, attn_override_mask)))
    
    inputs = tokenizer("Hi, I eat dog", return_tensors="pt")
    new_outputs = model(**inputs, output_hidden_states = True, output_attentions = True)

    for hook in hooks: hook.remove()

    inputs = tokenizer("Hi, I eat dog", return_tensors="pt")
    original_outputs = model(**inputs, output_hidden_states = True, output_attentions = True)

    new = new_outputs.attentions[-1][0, :3, :, :]
    old =  original_outputs.attentions[-1][0, :3, :, :]

    # print(old)
    # print(f"torch.all(new.eq(old)) : {torch.all(new.eq(old))}")
    print(f"==== original distribution of model") 
    print(F.softmax(original_outputs.logits, dim=-1))
    print(f"old prediction : {torch.argmax(original_outputs.logits, dim=-1)}")
    
    # print(new)
    print(f"==== new distribution of model") 
    print(F.softmax(new_outputs.logits, dim=-1))
    print(f"new prediction : {torch.argmax(new_outputs.logits, dim=-1)}")

breakpoint()

# Note:
# attention interventions attention heads αl,h ~ softmax 

"""
aim: override attention value and pass through entire models to get prediction 


override attention data:
        layer: specify layer
        attention_override:  values to override the computed attention weights
        attention_override_mask: which attention wieght to override

idea: manipulate and combine attention of all heads 
then forward thorugh enire model ?

hook current layer and providing value and postion ?

register forward hook: get output of particular module

attention_prob : alpha 
value_layer : value variable in attention paper
context_layer :  alpha @ value_layer

this is one isntance intervene and get result


# perform_intervention
      base_string: The doctor asked the nurse a question
      substitutes:  he, she
      candidates: asked, answer


get_prob_for_multi_token 
 context : x: # E.g. The doctor asked the nurse a question. She
 candidates : internvention_candidates_tok:  asked, answer
 return: prob of muliti-token in given context

attention_invervention_experiment:
    direct effect:
        input:  "===== He"
    indirect effect:
        input:  "===== she"
    
    get attention override:
        forward combine of (input, candiate)
            indirect effect The doctor asked the nurse a question. She

 idea2:
    1. collect attention called attention_override
    2. use to attention_override to replace inside forward condition 
    in normal forward then get new_prob distribution

attention_intervention -> new probilities
    candidate:  still the same concept as before ?
    hooks:  to collect return interven attention head of normal forward
    mlm_inputs ?
    use hooks to get new distribution ?
    why collect hooks for what ?

    guess: 
      1. hook register to manipulate module with attention weight we need 
      2. reald foward will use one manipulated by regsitering hook before



what kinds of input considered to be  attention_override  ?
what kinds of input considered to be  normal forward  ?
where do we get new distribution after intervene ?

"""
