import socket
import pickle
import sys
import torch
import os

from hospital.config import *
from hospital.model import CNN
from hospital.dataset import NPZDataset
from hospital.local_train import train
from hospital.prototype import build_prototypes
from hospital.logger import ClientLogger

hospital = sys.argv[1]
logger = ClientLogger(hospital)

ds = NPZDataset(f"fedpc_bloodmnist_npz/{hospital}/train.npz")
dl = torch.utils.data.DataLoader(ds, BATCH_SIZE, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN(len(ds.classes), device=device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
os.makedirs("local_models", exist_ok=True)

sock = socket.socket()
sock.connect((SERVER_HOST, SERVER_PORT))

# Handshake
sock.sendall(pickle.dumps(hospital))
ack = pickle.loads(sock.recv(1024))
if ack != "ACK":
    raise RuntimeError("Server handshake failed")

global_protos = {}

for r in range(ROUNDS):
    lam = min(LAMBDA_PROTO, 0.5 * (r + 1))
    logger.round_start(r, ROUNDS, lam)

    acc = train(model, dl, opt, global_protos, lam)
    logger.train_result(acc)

    model_path = f"local_models/{hospital}_round_{r}.pt"
    torch.save(model.state_dict(), model_path)
    logger.model_saved(model_path)

    # Build prototypes and log full weight detail before sending
    local_protos = build_prototypes(model, dl)
    logger.log_sending_protos(local_protos)

    sock.sendall(pickle.dumps(local_protos))

    # Receive and log full weight detail of what server returned
    global_protos = pickle.loads(sock.recv(10_000_000))
    logger.log_received_protos(global_protos)

    # Log per-class drift: how far did the server move each prototype?
    logger.log_proto_delta(local_protos, global_protos)

sock.close()
logger.training_complete()