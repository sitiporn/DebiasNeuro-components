# Debias Neuro-components leveraging Causal mediation analysis

### Todo
-  [x] intervention exist bias neurons by replace hidden representation of [CLS] with zero
-  [x] remove [CLS] representations of candidates neurons on the High-overlap score set 
-  [x] reformat model's predictions of both Null and Intervention to either entail or non entailment
-  [x] using package of challenge set's paper to evaluate model's predictions
-  [x] find top 20 K-percent neurons based on NIE scores
-  [x] maksing incremental top 1-20 K-percent neurons and computing heuristic
-  [x] plot the incremental heuristic scores between 1 and 20 
-  [ ] vary the number of neurons from topk from entire layers experiment on (MNLI-mismatched)
-  [ ] vary value of neuron's representation experiment on (MNLI-mismatched)
-  [ ] perform attention heads intervention using counterfactual and all set up same as neuron intervention

## Result

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
