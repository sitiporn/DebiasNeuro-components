from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch 
from utils import BertAttentionOverride
import torch.nn.functional as F

"""
mnli 

"""

# model fine-tune on mnli

model_name = 'sileod/roberta-base-mnli'
model = AutoModelForSequenceClassification(model_name)
config = AutoConfig.from_pretrained(model_name)

# intervention model's attention on 
# hans eval set


# Load hans set
