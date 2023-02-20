# Debias Neuro-components leveraging Causal mediation analysis

## Todo
- [ ] preediciton analysis 
    - [x] HOL and LOL set
    - [x] subsample used to compute indirect effect 
    - [x] analyze distributions

- [ ] changing subsample methods
    - [x] balancing subsamples set among classes  of NIE set
    - [x] balancing subsamples set among classes  of HOL and LOL set
    - [x] get rid of samples are not belong to 3 classes
    - [x] compute average prob of every samples for golden answer of each samples

- [ ] compute average embeddings of [CLS] of HOL and LOL set add feed to FFN to softmax to see distribution of each class

- [ ] changing y computation by consider non-entailment class to compute indirect effect

- [ ] further interventions
    - [ ] neuron intervention by replace set of of candidates from top K of each layer (hidden representation; output)
    - [ ] attention head interventions by changing attention weights of individual attention head; input
    - [ ] full layer of attention head; input; to the important of attention layers

## General Tackles
1. modifying embedding space to do counterfactual inference
2. modifying input and applying casual inference theory 
3. using interpretability technique to find meaning of what we intervene and to find candidates components
4. intervene output of activation replace feature value of [CLS] of original one (no-treatment) by treatment one 


## Interpretability

* Purpose 

1. to find tenative layers based on
    1.1 assumption if each attention most attention to word overlap heavily that is the most bias component 
as they are rely on shortcut word

* Tackles

1. Gradient Based method to find feature important
2. Interpret attention head 
    - the assumption; most will attention word overlap



## Problems
- confident scores of sample that we get is high among samples 


## Notes

- how to we select candiate set to test bias ?
    - is there statistic of model prediction incorrecly on high overlap ?


- block entire path from X - > Y; TE = direct effect, IDE = 0; in the context of representation
- do we need to find candidate sample that have high TE first ?


## Observation 

- NIE-all and NIE-sum; NIE-all << NIE-sum (componet working together); Nuerons
- NIE-all and NIE-sum; NIE-all ~ NIE-sum (component can working indiviudually); attention head
- ref: Causal Mediation Analysis for Interpreting Neural NLP:
