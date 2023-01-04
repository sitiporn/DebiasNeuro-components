import os 
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
                tokenizer, 
                sentence_with_:str, 
                substitutes: list, 
                candidates: list, 
                device = 'cpu') -> None:

        super()
        
        self.tokenizer = tokenizer
        # All the initial strings
        # First item should be neutral, others tainted ? 
        self.base_strings = [base_string.format(s) for s in substitutes]

        # Where to intervene
        # Text position ?
        self.position = base_string.split().index('{}')

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






