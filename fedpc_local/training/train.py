import torch.nn.functional as F

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y_local, _ in loader:
        x, y_local = x.to(device), y_local.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y_local)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
