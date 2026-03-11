import socket
import pickle
import sys
import torch
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

print(f"Connecting to server at {SERVER_HOST}:{SERVER_PORT}...")
sock = socket.socket()
sock.connect((SERVER_HOST, SERVER_PORT))
print("Connected to server!")
global_protos = {}
for r in range(ROUNDS):
    acc = train(model, dl, opt, global_protos, LAMBDA_PROTO)
    print(f"[{hospital}] Local Acc: {acc:.2f}%")

    local_protos = build_prototypes(model, dl)
    sock.sendall(pickle.dumps(local_protos))

    global_protos = pickle.loads(sock.recv(10_000_000))



    