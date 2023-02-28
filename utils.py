from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gc
#from fairseq.data.data_utils import collate_tokens

#def report_gpu():
#   print(torch.cuda.list_gpu_processes())
#    gc.collect()
#    torch.cuda.empty_cache()


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
    
    thresholds = {"High-overlap": None, "Low-overlap": None}

    df['overlap_scores'] = df['pair_label'].apply(get_overlap_score)

    overlap_scores = df['overlap_scores'] 

    thresholds["Low-overlap"]  = np.percentile(overlap_scores, lower_bound)
    thresholds["High-overlap"] = np.percentile(overlap_scores, upper_bound)

    return thresholds
    
def group_by_treatment(thresholds, overlap_score, gold_label):

    # note : we dont care about 
    if overlap_score >= thresholds["High-overlap"]:
        return "HOL"
    elif overlap_score <= thresholds["Low-overlap"]: 
        return "LOL"
    else:
        return "exclude"

def get_activation(layer, do, activation, is_averaged_embeddings, class_name = None):

  # the hook signature
  def hook(model, input, output):
    
    # print(f"layer : {layer}, do : {do}, {output.shape} ")

    if class_name is None:
    
        if layer not in activation[do].keys():

            if is_averaged_embeddings:
                
                activation[do][layer] = 0

            else:
                activation[do][layer] = []

        # grab representation of [CLS] then sum up

        if is_averaged_embeddings:

            activation[do][layer] += torch.sum(output.detach()[:,0,:], dim=0)

        else:
            
            activation[do][layer].extend(output.detach()[:,0,:])
    else:

        if layer not in activation[do][class_name].keys():

            if is_averaged_embeddings:
                
                activation[do][class_name][layer] = 0

            else:
                activation[do][class_name][layer] = []

        # grab representation of [CLS] then sum up

        if is_averaged_embeddings:

            activation[do][class_name][layer] += torch.sum(output.detach()[:,0,:], dim=0)

        else:
            activation[do][class_name][layer].extend(output.detach()[:,0,:])
  
  return hook

def collect_output_components(model, experiment_set, dataloader, tokenizer, DEVICE, layers, heads, is_averaged_embeddings):
    
    """ get hidden representation all neurons """
   
    hooks =  {"High-overlap" : None, "Low-overlap": None}

    # linear layer
    q_layer = lambda layer : model.bert.encoder.layer[layer].attention.self.query
    k_layer = lambda layer : model.bert.encoder.layer[layer].attention.self.key
    v_layer = lambda layer : model.bert.encoder.layer[layer].attention.self.value
    
    self_output = lambda layer : model.bert.encoder.layer[layer].attention.output
    intermediate_layer = lambda layer : model.bert.encoder.layer[layer].intermediate
    output_layer = lambda layer : model.bert.encoder.layer[layer].output

    # output of value vector multiply with weighted scored
    # self_attention = lambda layer : model.bert.encoder.layer[layer].attention.self

    # using for register
    q = {}
    k = {}
    v = {}
   
    ao = {}
    intermediate = {}
    out = {} 

    #dicts to store the activations
    q_activation = {}
    k_activation = {}
    v_activation = {}
  
    ao_activation = {}
    intermediate_activation = {}
    out_activation = {} 

    
    attention_data = {}        

    # dict to store  probabilities
    distributions = {}

    counter = 0

    batch_idx = 0

    for batch_idx, (sentences, labels) in enumerate(tqdm(dataloader, desc=f"Intervene_set_loader")):
        
        for idx, do in enumerate(tqdm(['High-overlap','Low-overlap'], desc="Do-overlap")):

        # if batch_idx == 2:
        #     print(f"stop batching at index : {batch_idx}")
        #     break

        # counter += len(pair_sentences['High-overlap'][0])
        
        # print(f"current : {counter}")

            if do not in ao_activation.keys():
                
                q_activation[do] = {}
                k_activation[do] = {}
                v_activation[do] = {}
                
                ao_activation[do] = {}
                intermediate_activation[do] = {}
                out_activation[do] = {} 
                
                distributions[do] = {} 
                
            
            if experiment_set.is_group_by_class:

                for class_name in sentences[do].keys():

                    if class_name not in q_activation[do].keys():
                
                            q_activation[do][class_name] = {}
                            k_activation[do][class_name] = {}
                            v_activation[do][class_name] = {}
                            
                            ao_activation[do][class_name] = {}
                            intermediate_activation[do][class_name] = {}
                            out_activation[do][class_name] = {} 
                            
                            distributions[do][class_name] = {} 
                    
                    premise, hypo = sentences[do][class_name]
    
                    pair_sentences = [[premise, hypo] for premise, hypo in zip(premise, hypo)]
    
                    inputs = tokenizer(pair_sentences, padding=True, truncation=True, return_tensors="pt")
    
                    inputs = {k: v.to(DEVICE) for k,v in inputs.items()} 
                    
                    # register forward hooks on all layers
                    for layer in layers:
                        
                        q[layer] = q_layer(layer).register_forward_hook(get_activation(layer, do, q_activation, is_averaged_embeddings, class_name=class_name))
                        k[layer] = k_layer(layer).register_forward_hook(get_activation(layer, do, k_activation, is_averaged_embeddings, class_name=class_name))
                        v[layer] = v_layer(layer).register_forward_hook(get_activation(layer, do, v_activation, is_averaged_embeddings, class_name=class_name))
                        
                        ao[layer] = self_output(layer).register_forward_hook(get_activation(layer, do, ao_activation, is_averaged_embeddings, class_name=class_name))
                        intermediate[layer] = intermediate_layer(layer).register_forward_hook(get_activation(layer, do, intermediate_activation, is_averaged_embeddings, class_name=class_name))
                        out[layer] = output_layer(layer).register_forward_hook(get_activation(layer, do, out_activation, is_averaged_embeddings, class_name=class_name))

                    with torch.no_grad():    

                        # get activatation
                        outputs = model(**inputs)
                        # outputs = model(**inputs[do])

                    del outputs
                    
                    #report_gpu()
    
                # detach the hooks
                    for layer in layers:
                        
                        q[layer].remove()
                        k[layer].remove()
                        v[layer].remove()
                        
                        ao[layer].remove()
                        intermediate[layer].remove()
                        out[layer].remove()
                
                    inputs = {k: v.to('cpu') for k,v in inputs.items()} 
                
        
        batch_idx += 1

    with open('../pickles/activated_components.pickle', 'wb') as handle:

        pickle.dump(q_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(k_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(v_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(ao_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(intermediate_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(out_activation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        pickle.dump(attention_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(counter, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(experiment_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataloader, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"save activate components done ! ")

def test_mask(neuron_candidates =[]):

    x  = torch.tensor([[ [1,2,3], 
                         [4,5,6]
                       ],
                       [
                         [1,2,3],
                         [4,5,6]                           
                       ],
                       [
                         [1,2,3],
                         [4,5,6]                           
                       
                       ]])

    neuron_ids = [0, 1, 2]

    # define used to intervene
    mask = torch.zeros_like(x, dtype= torch.bool) 
    
    # bz, seq_len, hidden_dim
    mask[:, 0, neuron_ids] = 1

    print(f"Before masking X")
    print(x)

    print(f" ===  mask list of neurons : {neuron_ids}  and [CLS] ====")
    print(mask)

    value = torch.tensor([11, 12, 13])[neuron_ids]
    value = value.repeat(x.shape[0], x.shape[1], 1)
    
    print(f"after masking X ")
    print(x.masked_scatter_(mask, value))

class Classifier(nn.Module):

    def __init__(self, model):
        super(Classifier, self).__init__()

        self.model = model
        self.pooler = model.bert.pooler
        self.dropout = self.model.dropout
        self.classifier = self.model.classifier

    def forward(self, last_hidden_state):

        pooled_output = self.pooler(last_hidden_state)
        
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)

        return logits

