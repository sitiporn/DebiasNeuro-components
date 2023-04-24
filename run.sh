#!/bin/bash

# optional arguments:
#   -h, --help            show this help message and exit
#   --layer LAYER         layers to intervene
#   --treatment TREATMENT
#                         high or low overlap
#   --analysis ANALYSIS   compute cma analysis
#   --top_k TOP_K         get top K analysis
#   --distribution DISTRIBUTION
#                         get top distribution
#   --embedding_summary EMBEDDING_SUMMARY
#                         get average embeddings
#   --get_counterfactual GET_COUNTERFACTUAL
#                         get average embeddings
#   --trace TRACE         tracing counterfactual
#   --debias DEBIAS       debias component
#   --get_prediction GET_PREDICTION
#                         get distributions
#   --dev_name DEV_NAME   optional filename
#   --weaken WEAKEN       best weaken rate towards activation
#   --neuron_group NEURON_GROUP
#                         best combination group of neurons to intervene


# python3 experiment.py --dev_name mismatched --treatment True --get_prediction True --top_k True
# python3 experiment.py --dev_name hans --treatment True --get_prediction True --top_k True --weaken 0.7936 --neuron_group 45
python3 experiment.py --dev_name  mismatched --treatment True --get_prediction True --top_k True  
