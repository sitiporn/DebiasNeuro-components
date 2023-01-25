# Debias Neuro-components leveraging Causal mediation analysis

## Todo
- [x] testing attention intervention
- [x] testing neurons intervention
- [x] grouping entailments by treatment
- [ ] naive approach on Indirect effect
    - [x] logging all neurons into pickle file
    - [x] compute average activations of [CLS]  from HOL set
    - [x] compute average activations of [CLS] from LOL set
    - [ ] compute causal effect between HOL set and HOL set replaced by average output neurons of LOL set 
    - [ ] compute causal effect between LOL set and LOL set replaced by average output neurons of HOL set
    - [ ] compute causal effect of all samples between original one and the one replaced by average output neurons of HOL set and  LOL set

- [ ] compute NIE and TE using replace the feature

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



