import os
import sys
import torch
from torch.utils.data import DataLoader

from config import *
from data.npz_dataset import NPZDataset
from models.cnn import CNN
from training.train import train_epoch
from training.eval import evaluate
from prototype.build import build_prototypes

def run_hospital(hospital):
    print(f"\n🏥 {hospital}")

    train_ds = NPZDataset(f"{DATA_ROOT}/{hospital}/train.npz")
    val_ds   = NPZDataset(f"{DATA_ROOT}/{hospital}/val.npz")
    test_ds  = NPZDataset(f"{DATA_ROOT}/{hospital}/test.npz")

    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(val_ds, BATCH_SIZE)
    test_ld  = DataLoader(test_ds, BATCH_SIZE)

    num_classes = len(train_ds.global_classes)
    model = CNN(num_classes).to(DEVICE)

    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val = 0
    os.makedirs(f"{SAVE_ROOT}/{hospital}", exist_ok=True)

    for ep in range(EPOCHS):
        loss = train_epoch(model, train_ld, optim, DEVICE)
        val_acc = evaluate(model, val_ld, DEVICE)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(),
                       f"{SAVE_ROOT}/{hospital}/backbone.pt")

        print(f"[{hospital}] Epoch {ep+1:02d} | Loss {loss:.3f} | Val {val_acc:.2f}%")

    model.load_state_dict(
        torch.load(f"{SAVE_ROOT}/{hospital}/backbone.pt")
    )

    test_acc = evaluate(model, test_ld, DEVICE)
    print(f"[{hospital}] ✅ Test Acc: {test_acc:.2f}%")

    protos = build_prototypes(model, train_ld, DEVICE)
    torch.save(protos, f"{SAVE_ROOT}/{hospital}/prototypes.pt")

    print(f"[{hospital}] 📦 Saved {len(protos)} prototypes")


if __name__ == "__main__":
    hospital = sys.argv[1]
    run_hospital(hospital)
