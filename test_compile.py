import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from learn_mask.sparse_trainer import SparseTrainer
from learn_mask.hf_model import SequenceClassificationTransformer
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from my_package.data import get_all_model_paths, get_all_checkpoints
from my_package.utils import load_model
import copy
import yaml

# Symtom : 
# 1. learning rate doesnot chage
# 2. performance huge drop  

# load model
seed = 1548
label_maps = {"entailment": 0, "contradiction": 1, "neutral": 2}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = 'configs/learn_mask/unstructured_sigmoid.yaml'

with open(config_path, "r") as yamlfile:
    sparse_args = yaml.load(yamlfile, Loader=yaml.FullLoader)

model_name  =  '../bert-base-uncased-mnli/'
# load transformer model
model = SequenceClassificationTransformer.from_pretrained(model_name, num_labels = 3)
sparse_args = SparseTrainingArguments(**sparse_args)
# patch model
model_patcher = ModelPatchingCoordinator(
        sparse_args= sparse_args,
        device=device,
        cache_dir="tmp/",  # Used only for teacher
        logit_names="logits",  # TODO
        teacher_constructor=None,  # TODO
)

model_patcher.patch_model(model)
model_patcher.schedule_threshold(True)

# LOAD_MODEL_PATH = '../models/compile_test'
LOAD_MODEL_PATH = '../models/test_cp_debias_mask_lr_0.0025_recent_baseline/'

"""
test_cp_debias_mask_lr_0.01_recent_baseline
test_cp_debias_mask_lr_0.025_recent_baseline
test_cp_debias_mask_lr_0.0025_recent_baseline
"""

checkpoints = get_all_checkpoints(LOAD_MODEL_PATH)

for checkpoint in checkpoints:
    # copy patch architecture
    cur_model = copy.deepcopy(model)
    # load patch model
    print(f'load model from : {checkpoint}')
    cur_model.load_state_dict(torch.load(checkpoint))
    # compile
    removed, heads = model_patcher.compile_model(cur_model)
    print(f"Compiled model. Removed {removed} / {heads} heads.")
    torch.save(cur_model.state_dict(), checkpoint)

    