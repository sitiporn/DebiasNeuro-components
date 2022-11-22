from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = 'bert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")


breakpoint()

# Note:
# attention interventions attention heads αl,h ~ softmax ?
# module type that are the gauge of particular layers
# Goal
# ---
# invernene specific layer and head
# 1. manipulate attention head 
# 2. recompute again 
# 3. forward  result through below  architecure to get prediction

"""
aim: override attention value and pass through entire models to get prediction 

context: text

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