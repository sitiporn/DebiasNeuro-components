# Debias Neuro-components leveraging Causal mediation analysis

## Todo
- [x] testing attention intervention
- [x] testing neurons intervention
- [x] grouping entailments by treatment
- [ ] NIE of individual neurons
    - [ ] use validation set samples 2000 by balancing entailment and non-entailment(neutral and contradiction) inference from splited validation set to get predictions
    - [ ] perform data analysis to get theshold word overlap 80 percent and 20 percent from splited validation set to get HOL and LOL set
    - [ ] balance samples of HOL and LOL set to compute average of [CLS] of every neuron across HOL and LOL set separately
    - [ ] Compute NIE from splited validate set (every word overlap score used)

- [ ] NIE of set of neurons
    - [ ] intervene each set of neurons in first and second half of each linear layer of each modules (Q, K, V, AO, I, O)



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

