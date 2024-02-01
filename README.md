# Debias Neuro-components leveraging Causal mediation analysis

### Todo

- [x] Checking Custom optimizer vs Original Optimizer
- [ ] Document CMA Module
- [ ] Document gradient unlearning



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





        

        
