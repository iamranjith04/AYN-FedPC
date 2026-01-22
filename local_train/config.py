import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "fedpc_bloodmnist_npz"
SAVE_ROOT = "local_outputs"

BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
