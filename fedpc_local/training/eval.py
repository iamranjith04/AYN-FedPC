import torch   # 🔑 YOU WERE MISSING THIS

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y_local, _ in loader:
            x, y_local = x.to(device), y_local.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y_local).sum().item()
            total += y_local.size(0)

    return 100.0 * correct / total
