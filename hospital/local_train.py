import torch
import torch.nn.functional as F
from hospital.config import EPOCHS

def train(model, loader, opt, global_protos=None, lam=0.0):
    device = next(model.parameters()).device
    model.train()
    
    total_correct = 0
    total_samples = 0
    
    for epoch in range(EPOCHS):
        epoch_correct = 0
        epoch_total = 0
        
        for x, yl, yg in loader:
            x = x.to(device)
            yl = torch.tensor(yl, device=device)
            
            opt.zero_grad()
            out, feat = model(x, return_feat=True)
            
            loss = F.cross_entropy(out, yl)
            
            if global_protos is not None and len(global_protos) > 0:
                proto_loss = 0.0
                count = 0
                
                for i, y in enumerate(yg):
                    y_int = int(y)
                    if y_int in global_protos:
                        proto_tensor = global_protos[y_int].to(device)
                        proto_loss += F.mse_loss(feat[i], proto_tensor)
                        count += 1
                
                if count > 0:
                    proto_loss = proto_loss / count
                    loss = loss + lam * proto_loss
            
            loss.backward()
            opt.step()
            
            preds = out.argmax(1)
            epoch_correct += (preds == yl).sum().item()
            epoch_total += yl.size(0)
        
        total_correct = epoch_correct
        total_samples = epoch_total
    
    acc = 100.0 * total_correct / total_samples
    return acc