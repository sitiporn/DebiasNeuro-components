import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from my_package.utils import Intervention

"""
---
 layers : how many layers
 neurons: how many neurons

"""
def neuron_intervention(rep, layers, neurons, position):

# Hook for changing representation during forward pass
	def intervention_hook(module, input, output, position, neurons):
		print(f"inside intervention hook")

		return None

	return None


intervention_modes = ""
model_name = 'bert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name)

# get number of neurons to intervene on
neurons = model.config.hidden_size 

#print(f"The number of neurons : {num_neurons}")

# First get position across batch

# args: context, ouputs, rep,  layers, neurons, position, intervention_type, alpha ?

# then, for each element get correct index w/ gather

# neurons ?
# slice ? 
# position ?
# order_dims method
# intervention_type: "replace"


# Overwrite values in the output
# First define mask where to overwrite
#  

# output ?
# layers ?


neuron_layer = lambda layer : model.bert.encoder.layer[layer].output

# Get the neurons to intervene on
neurons = torch.LongTensor(neurons) #.to(device)

# To account for swtiched dimensions in model internals:
# Default: [batch_size, seq_len, seq_len, hidden_dim]
order_dims = lambda a: a

# position -> intervention.position ? 

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_string = None
substitutes = None
candidates = None

# load string from 

intervention = Intervention(tokenizer = tokenizer, string_with_treatment = , stringt_w_o_treatment = )

# Todo : get examples

candidates = [ex.female_occupation_continuation, ex.male_occupation_continuation]
substitutes = [ex.female_pronoun, ex.male_pronoun]
intervention = Intervention(tokenizer, ex.base_string, substitutes, candidates)

# slice ?
position = intervention.position

#base_slice = order_dims(slice(None), position, slice(None))

breakpoint()

# base_string ?

