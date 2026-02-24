import os
import torch
from torch.utils.data import DataLoader
from hospital.dataset import NPZDataset
from hospital.model import CNN

BACKBONE_PATH = "artifacts/encoder.pt"

def evaluate_global(global_protos, device="cpu"):
    if not os.path.exists(BACKBONE_PATH):
        print("⚠️ Encoder not found — skipping global evaluation")
        return None

    model = CNN(num_classes=1)
    model.encoder.load_state_dict(
        torch.load(BACKBONE_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    correct, total = 0, 0

    for h in ["hospital_1", "hospital_2", "hospital_3"]:
        test_path = f"fedpc_bloodmnist_npz/{h}/test.npz"
        if not os.path.exists(test_path):
            continue
            
        ds = NPZDataset(test_path)
        dl = DataLoader(ds, batch_size=128)

        with torch.no_grad():
            for x, _, yg in dl:
                x = x.to(device)
                _, feats = model(x, return_feat=True)
                
                for f, y in zip(feats, yg):
                    y_int = int(y)
                    
                    if y_int not in global_protos:
                        continue
                    
                    best_sim = -float('inf')
                    best_class = None
                    
                    for class_id, proto in global_protos.items():
                        proto = proto.to(device)
                        sim = torch.cosine_similarity(
                            f.unsqueeze(0), 
                            proto.unsqueeze(0), 
                            dim=1
                        ).item()
                        
                        if sim > best_sim:
                            best_sim = sim
                            best_class = class_id
                    
                    if best_class == y_int:
                        correct += 1
                    total += 1

    if total == 0:
        return None
        
    return 100.0 * correct / total