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

Question
---
    when do we adjust sharpness value (mitigate paper)
    using bias model to reweigth once performing grid search



Sharpness control
---

C ~ control the strenght of bias removal
 - optimize it on validation set by minimizing Kl-div between
    for TIE_A;   Ya,x∗ and Ya,x  
        a - > X-> Y
        a - > *X-> Y
    
    for TE_model;  softmax(c* Y_A->Y) and Y_A->X->Y

    A; artifact whether there are high word overlap or not
    X; Text as input to the model
    X - model ->  Y

    when model have with and without treatment how much different 
        - basically trying to make model less effect with treatment

    c ~ the same shape as output's distribution

    Ya,x ~ ensemble model between main model and bias model


    total effect = Direct effect + indirect effect

# Sharpness control
    
TIE
---
indirect effect = total effect - dirrect effect 
                = Ya,x - Ya, x*
    * remove direct effect from bias by substraction
    * we want c to adjust the outcome of mediators that having no-treatment close to treatment
    
TE
---
 Causal query ~ what the prediction will be if use deep learning model instead of bias model 

 TE_model = main model - c * bias model
 c ~ readjust the strength of bias model 
 softmax(c ∗ YA→Y)

mitigation paper
----
1. training bias model
2. tuning c to have counterfactual(bias model) act 
    as deep learning model and reducing indirect effect from mediator(Text) carrying artifact pass through main model (X->Y)

    TIE; treatment is artifact; 
     u ~ uniform distribution
     Pm = u if  X = x*
     Pb = u if  A = a*
     For TIEA , similar to Niu et al. (2021),
     setting no-treatment constant u to c

    minimize kl-div ; Ya,x − Ya,x∗ =  log σ (Pb + Pm ) - log σ (Pb + c)
        - the objective is to have output's distributions of main models closely; mediators have no-treatment closing to treatment

        - get c making main model learn to have effect from artifact even without it
        
    TE; treatment is to use deep learning model

# Reweighting

- using bias a model to reweighting the training sets (MNLI) ?
    - using the confident of bias model to quantify which sample
        bias model outoput: high confident correlate with bias 
            - the answers are correct; high weight  for that samples
            - the answers are incorrect; low weight for that samples
