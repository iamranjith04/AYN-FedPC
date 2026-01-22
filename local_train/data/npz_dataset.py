import numpy as np
import torch
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.x = data["images"].astype("float32") / 255.0
        self.y_global = data["labels"].astype("int64")

        # 🔑 local label remapping
        self.global_classes = sorted(set(self.y_global.tolist()))
        self.g2l = {g: i for i, g in enumerate(self.global_classes)}
        self.l2g = {i: g for g, i in self.g2l.items()}

        self.y_local = np.array([self.g2l[y] for y in self.y_global])

    def __len__(self):
        return len(self.y_local)

    def __getitem__(self, idx):
        img = torch.tensor(self.x[idx]).permute(2, 0, 1)
        return (
            img,
            torch.tensor(self.y_local[idx]),   # local label
            torch.tensor(self.y_global[idx])   # global label
        )
