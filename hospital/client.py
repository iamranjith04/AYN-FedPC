import socket
import pickle
import sys
import torch
import os
import time

from hospital.config import *
from hospital.config import ROUNDS
from hospital.model import CNN
from hospital.dataset import NPZDataset
from hospital.local_train import train
from hospital.prototype import build_prototypes

hospital = sys.argv[1]
print(f"🏥 {hospital} started (ASYNC)")

ds = NPZDataset(f"fedpc_bloodmnist_npz/{hospital}/train.npz")
dl = torch.utils.data.DataLoader(ds, BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN(len(ds.classes), device=device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
os.makedirs("local_models", exist_ok=True)

print(f"Connecting to server at {SERVER_HOST}:{SERVER_PORT}...")
sock = socket.socket()
sock.connect((SERVER_HOST, SERVER_PORT))
print("Connected to server!")
global_protos = {}
for r in range(ROUNDS):
    acc = train(model, dl, opt, global_protos, LAMBDA_PROTO)
    print(f"[{hospital}] Local Acc: {acc:.2f}%")

    model_path = f"local_models/{hospital}_update_{r}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[{hospital}] 💾 Local model saved -> {model_path}")

    print(f"[{hospital}] 📡 Sending prototypes:")

    for cls, proto in local_protos.items():
        print(
            f"   class {cls} -> "
            f"shape {tuple(proto.shape)} "
            f"norm {proto.norm().item():.4f}"
        )

    print(f"[{hospital}] 📊 Model weight summary:")

    for name, param in model.named_parameters():
        print(
            f"   {name} | mean={param.data.mean():.4f} "
            f"std={param.data.std():.4f}"
        )

    local_protos = build_prototypes(model, dl)
    sock.sendall(pickle.dumps(local_protos))

    global_protos = pickle.loads(sock.recv(10_000_000))



    