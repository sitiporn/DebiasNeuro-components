# Debias Neuro-components leveraging Causal mediation analysis

## Todo
- [x] NIE of individual neurons
    - [x] computing theshold scores using data from whole set of validation to divided into HOL and LOL set
    - [x] balances classs's HOL and LOL set to get average hidden representation outputed by neurons
    - [x] subsample uniformly distributed from validation set used to compute  NIE scores
    - [x] ranking top-k of NIE's neurons 
    - [x] get top-k of HOL and LOL of individual neuron intervention

- [ ] preediciton analysis 
    - [ ] HOL and LOL set
    - [ ] subsample used to compute indirect effect 
    - [ ] analyze distributions

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




