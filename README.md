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
-  [ ] roll out variable save into pickles
-  [ ] traning main model with reweighted loss  on canidate parameters 
-  [ ] performing EDA on MNNLI-matched set

## Result

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

