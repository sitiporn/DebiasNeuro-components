import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
#from fairseq.data.data_utils import collate_tokens


class BertAttentionOverride(nn.Module):
    """A copy of `modeling_bert.BertSelfAttention` class, but with overridden attention values"""

    def __init__(self, module, attn_override, attn_override_mask):
        """
        Args:
            module: instance of modeling_bert.BertSelfAttentionOverride
                from which variables will be copied
            attn_override: values to override the computed attention weights.
                Shape is [bsz, num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention weights to override.
                Shape is [bsz, num_heads, seq_len, seq_len]
        """
        super().__init__()
        # if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads)
        #     )
        self.output_attentions = True #module.output_attentions
        self.num_attention_heads = module.num_attention_heads
        self.attention_head_size = module.attention_head_size
        self.all_head_size = module.all_head_size
        self.query = module.query
        self.key = module.key
        self.value = module.value
        self.dropout = module.dropout
        # Set attention override values
        self.attn_override = attn_override
        self.attn_override_mask = attn_override_mask

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Intervention:
        # attn_override and attn_override_mask are of shape (batch_size, num_heads, override_seq_len, override_seq_len)
        # where override_seq_len is the length of subsequence for which attention is being overridden
        override_seq_len = self.attn_override_mask.shape[-1]

        # print("=====  Before : ")
        # print(attention_probs[:, :, :override_seq_len, :override_seq_len])
        attention_probs[:, :, :override_seq_len, :override_seq_len] = torch.where(
            self.attn_override_mask,
            self.attn_override,
            attention_probs[:, :, :override_seq_len, :override_seq_len])
        
        # print("=====  After : ")
        # print(attention_probs[:, :, :override_seq_len, :override_seq_len])
        
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        return outputs


def get_overlap_score(pair_label):

    prem_words = []
    hyp_words = []

    premise = pair_label[0].strip()
    hypothesis = pair_label[1].strip()
    gold_label = pair_label[2].strip()

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)

    count = 0
    for word in hyp_words:
        if word in prem_words:
            count+=1

    overlap_score = count/len(hyp_words)        

    return overlap_score

class Intervention():
    """Wrapper all possible interventions """
    def __init__(self, 
                encode, 
                premises: list, 
                hypothesises: list, 
                device = 'cpu') -> None:

        super()
        
        self.encode = encode
        self.premises = premises
        self.hypothesises = hypothesises
        
        self.pair_sentences = []
        
        for premise, hypothesis in zip(self.premises, self.hypothesises):

            # Encode a pair of sentences and make a prediction
            self.pair_sentences.append([premise, hypothesis])

        # Todo : sort text before encode to reduce  matrix size of each batch
        # self.batch_tok = collate_tokens([self.encode(pair[0], pair[1]) for pair in self.batch_of_pairs], pad_idx=1)
        # self.batch_tok = self.encode(self.premises, self.hypothesises,truncation=True, padding="max_length")

        """
        # All the initial strings
        # First item should be neutral, others tainted ? 
        
        self.base_strings = [base_string.format(s) for s in substitutes]

        # Where to intervene
        # Text position ?
        self.position = base_string.split().index('{}')
        """

def get_overlap_thresholds(df, upper_bound, lower_bound):
    
    thresholds = {"treatment": None, "no-treatment": None}

    df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)

    # Todo: get overlap_score for whole entailment sets
    entail_mask = (df['gold_label'] == "entailment").tolist()
    overlap_scores = df['overlap_scores'][entail_mask]

    thresholds["no-treatment"] = np.percentile(overlap_scores, lower_bound)
    thresholds["treatment"] = np.percentile(overlap_scores, upper_bound)

    return thresholds
    
def group_by_treatment(thresholds, overlap_score, gold_label):
    
    if gold_label == "entailment":
        if overlap_score >= thresholds['treatment']:
            return "treatment"
        elif overlap_score <= thresholds['no-treatment']:              
            return "no-treatment"
        else:
            return "exclude"
    else:
        return "exclude"


def get_activation(layer, do, activation):
  
  # the hook signature
  def hook(model, input, output):
    
    if layer not in activation[do].keys():
        activation[do][layer] = []

    activation[do][layer].append(output.detach())
  
  return hook



def collect_output_components(model, dataloader, tokenizer, DEVICE):
   
    hooks =  {"do-treatment" : None, "no-treatment": None}
    
    # Todo: generalize for every model 
    layers = [*range(0, 12, 1)]
    heads = [*range(0, 12, 1)]
    
    
    # AO = lambda layer : model.bert.encoder.layer[layer].output.dense 
    # get attention weight output


    # self_attention = lambda layer : model.bert.encoder.layer[layer].attention.self
    self_output = lambda layer : model.bert.encoder.layer[layer].attention.output
    intermediate_layer = lambda layer : model.bert.encoder.layer[layer].intermediate
    output_layer = lambda layer : model.bert.encoder.layer[layer].output


    # using for register
    ao = {}
    intermediate = {}
    out = {} 

    #dicts to store the activations
    ao_activation = {}
    intermediate_activation = {}
    out_activation = {} 
    attention_data = {}        

    inputs = {}

    batch_idx = 0

    for pair_sentences , labels in tqdm(dataloader):


        for do in ['do-treatment','no-treatment']:

            if do not in ao_activation.keys():
                ao_activation[do] = {}
                intermediate_activation[do] = {}
                out_activation[do] = {} 
            
            pair_sentences[do] = [[premise, hypo] for premise, hypo in zip(pair_sentences[do][0], pair_sentences[do][1])]
            inputs[do] = tokenizer(pair_sentences[do], padding=True, truncation=True, return_tensors="pt")

            inputs[do] = {k: v.to(DEVICE) for k,v in inputs[do].items()}

            
            # get attention weight 
            outputs = model(**inputs[do], output_attentions = True)

            logits = outputs.logits
            # labels 0: contradiction, 1: entailment, 2: neutral
            # predictions = logits.argmax(dim=1)
            # print(f"current prediction of {do} : {predictions[:10]}")
            # print(f"current labels {do} : {labels[do][:10]}")

            if do not in attention_data.keys():
                attention_data[do] = [outputs.attentions]
            else:
                attention_data[do].append(outputs.attentions)

            # register forward hooks on all layers
            for layer in layers:

                ao[layer] = self_output(layer).register_forward_hook(get_activation(layer, do, ao_activation))
                intermediate[layer] = intermediate_layer(layer).register_forward_hook(get_activation(layer, do, intermediate_activation))
                out[layer] = output_layer(layer).register_forward_hook(get_activation(layer, do, out_activation))

            # get activatation
            outputs = model(**inputs[do])
 
            # detach the hooks
            for layer in layers:
                ao[layer].remove()
                intermediate[layer].remove()
                out[layer].remove()

        
        batch_idx += 1



    with open('../pickles/activated_components.pickle', 'wb') as handle:

        pickle.dump(ao_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(intermediate_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(out_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(attention_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"save activate components done ! ")




def neuron_intervention(model,
                        tokenizer,
                        layers,
                        neurons,
                        dataloader,
                        intervention_type='replace'):

        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output):

            # overwrite value in the output
            # define mask to overwrite
            scatter_mask = torch.zeros_like(output, dtype = torch.bool)
            
            print(f"set of neurons to intervene {neurons}")
            
            # where to intervene
            scatter_mask[:,:, neurons] = 1

            # value to replace : (seq_len, batch_size, output_dim)
            value = torch.zeros_like(output, dtype = torch.float)

            # (bz, seq_len, input_dim) @ (input_dim, output_dim)
            #  seq_len, batch_size, hidden_dim 
            
        neuron_layer = lambda layer : model.model.encoder.sentence_encoder.layers[layer].final_layer_norm
        
        handle_list = []     

        for batch_idx, (pair_sentences, label) in enumerate(dataloader):

            print(f"batch_idx : {batch_idx}")
            print(f"current pair sentences : {type(pair_sentences)}")
            print(f"current label : {label}")

            # if isinstance(pair_sentences[0], tuple)
            pair_sentences = [[premise,hypo] for  premise, hypo in zip(pair_sentences[0],pair_sentences[1])]

            # Compute token embeddings
            with torch.no_grad():

                input = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
                output = model(**input)
                logits = output.logits
                
                # labels 0: contradiction, 1: entailment, 2: neutral
                predictions = logits.argmax(dim=1)
                print(f"current prediction : {predictions}")

            for layer in layers:
                handle_list.append(neuron_layer(layer).register_forward_hook(intervention_hook))

            new_logprobs = model.predict('mnli', intervention.batch_tok[0:8])
            predictions = new_logprobs.argmax(dim=1)
        
            print(f"=== with intervene ====")
            print(new_logprobs[:8,:])
            print(predictions)

            for hndle in handle_list:
                hndle.remove() 
            
            logprobs = model.predict('mnli', intervention.batch_tok[0:8])
            predictions = logprobs.argmax(dim=1)
            
            print(f"=== without intervene ====")
            print(logprobs[:8,:])
            print(predictions)




"""
neuron_intervention
- context
- outputs
- rep
- layers
- neurons
- position
- intervention_type
- alpha

intervention_hook:
- pos : input, output
- alternative : positiion, neurons, intervention, intervention_type

others:
- base_string: eg. doctor said that {}
- substitutes : male or female pronoun 
- candidates : occupation of male and female : list 
- position: index of word that is last position of base_string that will be replaced
    * we dont use because we considering word overlap decomposed into high_word_overlap, low_word_overlap

- context; eg. the teacher said that 
- intervention.condidates_tok
- rep
- layer_to_search
- neuron_to_search
- interventoin.position

base_sentence = "The {} said that"
biased_word = "teacher"
intervention = Intervention(
        tokenizer,
        base_string = base_sentence,
        substitutes = [biased_word, "man", "woman"],
        candidates = ["he", "she"],
        device=DEVICE)


interventions = {biased_word: intervention}

"""
