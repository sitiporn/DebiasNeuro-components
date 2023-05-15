# Masking as an Efficient Alternative to Finetuning
for Pretrained Language Models

Wl ∈ {Wl_K , Wl_Q, Wl_V , Wl_AO, Wl_I , Wl_O}

WP and WT are always masked

they mask top-down and bottom up approach with sparsity %5 quite good in Fig2

ref- https://aclanthology.org/2020.emnlp-main.174.pdf

# Debiasing Masks A New Framework for Shortcut Mitigation in NLU

 - pruning rate of layer 2-4 has high 

# hypothesis

do they mask in sequential to let real mask learn because real mask associated with real weight ?

or we understand the shape of linear layer weight wrong ?

discard weight : weight zero

problems:
--------
result: run time usage for searching through all neurons to localize the position of neurons is endlessly long


issue:
-----

finding neurons position working on annotation artifact or at least potential candiates at first place ?


Question
---

how they inference while performing neuron intervention



#Experiment

individual neurons
----
1. use validation set samples 2000 by balancing entailment and non-entailment(neutral and contradiction)
2. inference from splited validation set to get predictions
3. perform data analysis to get theshold word overlap 80 percent and 20 percent from splited validation set to get HOL and LOL set
4. balance samples of HOL and LOL set to compute average of [CLS] of every neuron across HOL and LOL set separately
5. Compute NIE from splited validate set (every word overlap score used)

set of neurons
---
each layer

intervene each set of neurons in first and second half of each linear layer of each modules (Q, K, V, 
AO(attention output), I(intermediate of FFN), O(output of FFN))


## Research problem

problem Feb 13
---

hypothesis: 
    1. HOL NIE should be increased
    2. LOL NIE will should be decreased


what does it mean LOL if NIE still increase 

Why NIE of still 


## Note

1. when do we use 
