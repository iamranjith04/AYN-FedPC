import socket
import pickle
import threading
import os
import torch

from server.config import *
from server.aggregator import AsyncAggregator

os.makedirs("global_models", exist_ok=True)

aggregator = AsyncAggregator(N_CLASSES, ASYNC_ALPHA)

def handle_client(conn, addr):
    print(f"🔗 Connected: {addr}")

    while True:
        try:
            raw = conn.recv(10_000_000)
            if not raw:
                break

            local_protos = pickle.loads(raw)

            aggregator.update(local_protos)

            global_protos = aggregator.get_global()
            conn.sendall(pickle.dumps(global_protos))

            print(f"📦 Updated from {addr}")

            if aggregator.update_count % SAVE_EVERY == 0:
                torch.save(
                    aggregator.get_global(),
                    f"global_models/async_checkpoint_{aggregator.update_count}.pt"
                )
                print("💾 Checkpoint saved")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            break

    conn.close()
    print(f"❌ Disconnected: {addr}")


print("🚀 ASYN-FEDPC Server Started")

sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.listen()

while True:
    conn, addr = sock.accept()
    threading.Thread(
        target=handle_client,
        args=(conn, addr),
        daemon=True
    ).start()