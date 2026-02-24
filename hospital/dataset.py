import numpy as np, torch
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)
        self.x = d["images"]/255.0
        self.yg = d["labels"]
        self.classes = sorted(set(self.yg))
        self.map = {c:i for i,c in enumerate(self.classes)}
        self.yl = [self.map[y] for y in self.yg]

    def __len__(self): return len(self.yl)

    def __getitem__(self,i):
        x = torch.tensor(self.x[i]).permute(2,0,1).float()
        return x, self.yl[i], self.yg[i]
