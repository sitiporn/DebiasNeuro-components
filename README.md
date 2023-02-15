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
