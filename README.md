# Debias Neuro-components leveraging Causal mediation analysis

### Todo
-  [x] intervention exist bias neurons by replace hidden representation of [CLS] with zero
-  [x] remove [CLS] representations of candidates neurons on the High-overlap score set 
-  [x] reformat model's predictions of both Null and Intervention to either entail or non entailment
-  [x] using package of challenge set's paper to evaluate model's predictions
-  [x] find top 20 K-percent neurons based on NIE scores
-  [x] maksing incremental top 1-20 K-percent neurons and computing heuristic
-  [x] plot the incremental heuristic scores between 1 and 20 
-  [x] vary the number of neurons from topk from entire layers experiment on (MNLI-mismatched)
-  [x] vary value of neuron's representation experiment on (MNLI-mismatched)
-  [x] vary value 0.1, 0.2 -> 1 after that divide it down until get best val 
-  [x] adjust sharpness value (mitigate paper)
-  [x] using bias model to reweigth once performing grid search
-  [x] save train and untrain components used to train partition 
-  [x] roll out variable save into pickles
-  [x] recheck intervention position inside model
-  [x] recheck reweight set up using MNLI-matched-dev ?
-  [x] traning main model with reweighted loss on canidate parameters and plot losses
-  [x] uncomment load intitial partition parameters
-  [x] write tester to check parameters which are candidate-only weights to optimize
-  [x] Custom Auto gradients to perform specific value brackpropagation  
-  [x] Scheduler computation get accurate 
-  [ ] compute counterfactual for all seeds
-  [ ] finding candidate parameters from recent models (an existing function) using data for computing NIE
-  [ ] Perform gradient reversal (backward optimization)
-  [ ] Perform EDA on MNNLI-matched set to select to advantaged samples(preferred by bias model)
-  [ ] Train on candidate parameters using training set utilizing existing partition parameter function

## Open questions

## General Tackles
1. modifying embedding space to do counterfactual inference
2. modifying input and applying casual inference theory 
3. using interpretability technique to find meaning of what we intervene and to find candidates components
4. intervene output of activation replace feature value of [CLS] of original one (no-treatment) by treatment one 

## Observation 

- NIE-all and NIE-sum; NIE-all << NIE-sum (componet working together); Nuerons
- NIE-all and NIE-sum; NIE-all ~ NIE-sum (component can working indiviudually); attention head
- ref: Causal Mediation Analysis for Interpreting Neural NLP:


## Research Question

- Which components mediate annotation artifact [NIE] ?
    - [Layers/ Neurons] NIE of top 5% of neurons in each layers
    - [Attention head]
    - [Neurons overhead] across different cases

- Counterfactual and representation
    1. High overlap word overlap [Avg rep]
        - all above threshold 
        - control for class balance overlap vs non-overlap
    2. Low overlap word overlap  [Avg rep]
        - all above threshold 
        - control for class balance overlap vs non-overlap

- Things we can control  condition on: 
    - Class 
    - Annotation artifact
    - etc.
## Debias Technique on Training set

1. PCGU : They used the same set(Winogender) to 
  1.1 find (calculate constrastive gradients) bias sources by ranking gradients similarity
  1.2 tune masked langugage models on 
    advantage pair (a1): 
     1. often coreferent 
     2. more preffered by bias model
     3. reweight perspective: low scalers -> lower loss to update main model parameters
        
    disadvantage pair (a2):
     1. less often coreferent 
     2. less preferred by bias model
     3. reweight perspective: high scalers -> higher loss to update main model parameters

*** optimize step:
   - decreasing maximal probability of advantaged term

2. Ours:
   1. Calculate NIE scores on (valid set) multinli_1.0_dev_matched.jsonl to find bias neurons
   2. training set to tune models (masked language models) 
     2.1 perform EDA ~ to find representative threshold for a1 and a2 
     2.2 decreasing maximal probability of advantaged and increase the probability of disadvantaged 
     2.3 components: 
         In encoder: Update bias-only parameters, the rest is frozen
         outside encoder's components are also updated eg. classifier, pooler, embeddings are also update     

Ours vs PCGU using resources:
---
  1. ranking bias sources on 
     - valid set vs training set
     - using trained model ishan/bert-base-uncased-mnli  vs masked language models
  2. optimize masked language model on 
     - training set vs traing set



        

        
