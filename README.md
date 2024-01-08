# Debias Neuro-components leveraging Causal mediation analysis

### Todo
- [x] Perform EDA on MNNLI-matched set to select to advantaged samples(preferred by bias model)
- [x] recheck model(ishan/bert-base-uncased-mnli) weaken rate noted in config testing on dev-mm and HANS to find where a result increases
* checking neuron positions between High and Low overlap treatment if they are the same
  * same entail neurons position
  * bugs in some components 
    * nie shouldnt be the same because different representations
    * checking representation values between two components
*  
* implement group by class in get_hidden_representations
* extend counterfactual set from validation to training set

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



        

        
