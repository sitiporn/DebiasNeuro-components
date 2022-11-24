from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
from utils import BertAttentionOverride
import torch.nn.functional as F




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
    # AttributeError: 'BertSelfAttention' object has no attribute 'output_attentions'                                                                           
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
"""
==== new's output distribution of model
tensor([[0.4183, 0.5817]])
new prediction : tensor([1])
old's output distribution of model
tensor([[0.4234, 0.5766]])
old  prediction : tensor([1])


==== new's output distribution of model
tensor([[0.5593, 0.4407]])
new prediction : tensor([0])
old's output distribution of model
tensor([[0.5593, 0.4407]])
old  prediction : tensor([0])


"""


"""
===== new distribution ======
tensor([[[0.5985, 0.0314, 0.0645, 0.0533, 0.0170, 0.0192, 0.2161],
         [0.0417, 0.0113, 0.0169, 0.0107, 0.0086, 0.0098, 0.9011],
         [0.0621, 0.0294, 0.0267, 0.0121, 0.0112, 0.0340, 0.8245],
         [0.0155, 0.0074, 0.0116, 0.0099, 0.0067, 0.0046, 0.9443],
         [0.0072, 0.0025, 0.0026, 0.0029, 0.0107, 0.0044, 0.9696],
         [0.0548, 0.0086, 0.0136, 0.0160, 0.0172, 0.0336, 0.8562],
         [0.0146, 0.0042, 0.0121, 0.0089, 0.0047, 0.0050, 0.9507]],

        [[0.4326, 0.0810, 0.0763, 0.1192, 0.1338, 0.0653, 0.0919],
         [0.0676, 0.0118, 0.0131, 0.0121, 0.0081, 0.0318, 0.8555],
         [0.0564, 0.0222, 0.0266, 0.0404, 0.0183, 0.0207, 0.8155],
         [0.0180, 0.0066, 0.0176, 0.0227, 0.0160, 0.0186, 0.9006],
         [0.0284, 0.0074, 0.0219, 0.0128, 0.0075, 0.0272, 0.8948],
         [0.0783, 0.1033, 0.2421, 0.1843, 0.0672, 0.0871, 0.2376],
         [0.0167, 0.0038, 0.0162, 0.0237, 0.0119, 0.0196, 0.9081]],

        [[0.1318, 0.1509, 0.0746, 0.2492, 0.1124, 0.2582, 0.0229],
         [0.0637, 0.2396, 0.0854, 0.0621, 0.0462, 0.2091, 0.2940],
         [0.1238, 0.1358, 0.1247, 0.1998, 0.0973, 0.2051, 0.1135],
         [0.0535, 0.1134, 0.1048, 0.1490, 0.0737, 0.1194, 0.3862],
         [0.0765, 0.0462, 0.0397, 0.0365, 0.0374, 0.0720, 0.6918],
         [0.0481, 0.0489, 0.0402, 0.0986, 0.0878, 0.0754, 0.6011],
         [0.0350, 0.0166, 0.0311, 0.0172, 0.0171, 0.0262, 0.8568]]])
===== original distribution ======
tensor([[[0.3009, 0.0189, 0.0518, 0.0497, 0.0282, 0.0256, 0.5249],
         [0.0206, 0.0072, 0.0171, 0.0092, 0.0065, 0.0079, 0.9314],
         [0.0289, 0.0218, 0.0279, 0.0115, 0.0099, 0.0275, 0.8725],
         [0.0086, 0.0052, 0.0111, 0.0080, 0.0055, 0.0038, 0.9577],
         [0.0032, 0.0014, 0.0020, 0.0020, 0.0061, 0.0027, 0.9827],
         [0.0248, 0.0050, 0.0120, 0.0117, 0.0137, 0.0249, 0.9080],
         [0.0089, 0.0033, 0.0101, 0.0073, 0.0043, 0.0041, 0.9620]],

        [[0.3967, 0.0588, 0.0726, 0.1125, 0.1389, 0.0623, 0.1582],
         [0.0379, 0.0088, 0.0145, 0.0104, 0.0077, 0.0314, 0.8893],
         [0.0400, 0.0193, 0.0274, 0.0336, 0.0203, 0.0235, 0.8359],
         [0.0152, 0.0059, 0.0207, 0.0198, 0.0146, 0.0213, 0.9025],
         [0.0208, 0.0052, 0.0219, 0.0121, 0.0064, 0.0277, 0.9060],
         [0.0721, 0.0783, 0.2494, 0.1590, 0.0621, 0.1042, 0.2748],
         [0.0114, 0.0040, 0.0174, 0.0202, 0.0111, 0.0196, 0.9163]],

        [[0.1895, 0.1381, 0.0892, 0.1514, 0.0910, 0.2254, 0.1154],
         [0.0674, 0.1843, 0.0558, 0.0425, 0.0356, 0.1162, 0.4982],
         [0.1810, 0.1197, 0.1103, 0.1565, 0.0884, 0.1882, 0.1558],
         [0.0682, 0.0724, 0.0873, 0.1075, 0.0504, 0.0925, 0.5217],
         [0.0649, 0.0306, 0.0267, 0.0225, 0.0230, 0.0406, 0.7917],
         [0.0516, 0.0290, 0.0365, 0.0615, 0.0499, 0.0590, 0.7125],
         [0.0263, 0.0142, 0.0301, 0.0159, 0.0169, 0.0231, 0.8735]]])

"""

    











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