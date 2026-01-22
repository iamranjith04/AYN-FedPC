import torch
from collections import defaultdict

def build_prototypes(model, loader, device):
    model.eval()
    bucket = defaultdict(list)

    with torch.no_grad():
        for x, _, y_global in loader:
            x = x.to(device)
            _, feats = model(x, return_features=True)

            for f, g in zip(feats, y_global):
                bucket[int(g.item())].append(f.cpu())

    return {k: torch.stack(v).mean(0) for k, v in bucket.items()}
