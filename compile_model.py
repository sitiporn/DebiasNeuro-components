import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from learn_mask.sparse_trainer import SparseTrainer
from learn_mask.hf_model import SequenceClassificationTransformer
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from my_package.data import get_all_model_paths, get_all_checkpoints
from my_package.utils import load_model
import copy
import yaml
import argparse

# Symtom : 
# 1. learning rate doesnot chage
# 2. performance huge drop  

parser = argparse.ArgumentParser()
# parser.add_argument("--seed", type=int, help="The random seed value")	
parser.add_argument("--dataset_name", type=str, help="dataset name to read config") 


args = parser.parse_args()
print(args)

if args.dataset_name == 'mnli':
    config_path = 'configs/learn_mask/train_config_mnli.yaml'
elif args.dataset_name == 'qqp':
    config_path = 'configs/learn_mask/train_config_qqp.yaml'
elif args.dataset_name == 'fever':
    config_path = 'configs/learn_mask/train_config_fever.yaml'
else: 
    raise Exception("Sorry, no given dataset name")

with open(config_path, "r") as yamlfile: config = yaml.load(yamlfile, Loader=yaml.FullLoader)

label_maps = config['label_maps']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# seed = args.seed


config_path = 'configs/learn_mask/unstructured_sigmoid.yaml'
with open(config_path, "r") as yamlfile:
    sparse_args = yaml.load(yamlfile, Loader=yaml.FullLoader)

model_name  =  config['model']['model_name']
# load transformer model
model = SequenceClassificationTransformer.from_pretrained(model_name, num_labels = len(label_maps))
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
# model to compile
# LOAD_MODEL_PATH = f'../models/rm_debias_mask_lr_0.0025_recent_baseline/'

# LOAD_MODEL_PATH = f'../models/cp_debias_mask_lr_0.0025_recent_baseline/'
# LOAD_MODEL_PATH = '../models/test_cp_debias_mask_lr_0.0025_recent_baseline/'
# LOAD_MODEL_PATH = '../models/debias_mask_lr_0.0025_recent_baseline_clark/'
# LOAD_MODEL_PATH = '../models/debias_mask_lr_0.0025_baseline_qqp_mysplit/'
LOAD_MODEL_PATH = '../models/debias_mask_lr_0.0025_recent_baseline_clark/'

"""
test_cp_debias_mask_lr_0.01_recent_baseline
test_cp_debias_mask_lr_0.025_recent_baseline
test_cp_debias_mask_lr_0.0025_recent_baseline
"""

checkpoints = get_all_checkpoints(LOAD_MODEL_PATH)

cur_point = []

for checkpoint in checkpoints:
    split_path = checkpoint.split('/')[:-1]
    split_path.append('compile_model.bin')
    save_path =  '/'.join(split_path)
    # copy patch model architecture
    cur_model = copy.deepcopy(model)
    # # load patch model
    print(f'load_path : {checkpoint}')
    cur_model.load_state_dict(torch.load(checkpoint))
    # # # compile
    removed, heads = model_patcher.compile_model(cur_model)
    print(f"Compiled model. Removed {removed} / {heads} heads.")
    print(f'save_path : {save_path}')
    
    # save compile model
    torch.save(cur_model.state_dict(), save_path )

"""
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3990/checkpoint-140000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3990/checkpoint-140000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3990/checkpoint-70000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3990/checkpoint-70000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_409/checkpoint-140000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_409/checkpoint-140000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_409/checkpoint-60000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_409/checkpoint-60000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_1548/checkpoint-100000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_1548/checkpoint-100000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_1548/checkpoint-60000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_1548/checkpoint-60000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3099/checkpoint-140000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3099/checkpoint-140000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3099/checkpoint-70000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3099/checkpoint-70000/compile_model.bin
******
path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3785/checkpoint-140000/pytorch_model.bin
new path: ../models/debias_mask_lr_0.0025_recent_baseline_clark/seed_3785/checkpoint-140000/compile_model.bin
"""