import socket
import pickle
import threading
import os
import torch

from server.config import *
from server.aggregator import AsyncAggregator
from server.cluster import cluster_prototypes
from server.logger import ServerLogger

os.makedirs("global_models", exist_ok=True)

aggregator = AsyncAggregator(N_CLASSES, ASYNC_ALPHA)
logger = ServerLogger()

MIN_HOSPITALS_FOR_CLUSTERING = 2


def build_response(hospital_id):
    """
    Returns (response_protos, cluster_id_or_None).
    cluster_id is None when falling back to global protos.
    """
    global_protos = aggregator.get_global()

    if aggregator.n_hospitals() < MIN_HOSPITALS_FOR_CLUSTERING:
        return global_protos, None

    all_hospital_protos = aggregator.get_all_hospital_protos()

    try:
        _, hospital_to_cluster, cluster_class_protos = cluster_prototypes(
            all_hospital_protos, n_clusters=N_CLUSTERS
        )
    except Exception as e:
        logger._log(f"   ↳ clustering failed ({e}), falling back to global protos")
        return global_protos, None

    logger.clustering_result(hospital_to_cluster)

    cluster_id = hospital_to_cluster.get(hospital_id)
    if cluster_id is None:
        return global_protos, None

    merged = {**global_protos, **cluster_class_protos[cluster_id]}
    return merged, cluster_id


def handle_client(conn, addr):
    try:
        raw_id = conn.recv(1024)
        hospital_id = pickle.loads(raw_id)
        conn.sendall(pickle.dumps("ACK"))
        logger.client_connected(hospital_id, addr)
    except Exception as e:
        logger._log(f"⚠️  Handshake failed from {addr}: {e}")
        conn.close()
        return

    while True:
        try:
            raw = conn.recv(10_000_000)
            if not raw:
                break

            local_protos = pickle.loads(raw)

            aggregator.update(hospital_id, local_protos)

            # Log what was received with full weight detail
            logger.log_received_protos(hospital_id, local_protos, aggregator.update_count)

            # Log current global state after this update
            logger.log_global_state(aggregator.get_global())

            # Build and log personalised response
            response, cluster_id = build_response(hospital_id)
            logger.log_sending_protos(hospital_id, response, cluster_id)

            conn.sendall(pickle.dumps(response))

            # Atomic checkpoint
            if aggregator.update_count % SAVE_EVERY == 0:
                snapshot = aggregator.get_global()
                tmp   = f"global_models/async_checkpoint_{aggregator.update_count}.pt.tmp"
                final = f"global_models/async_checkpoint_{aggregator.update_count}.pt"
                torch.save(snapshot, tmp)
                os.replace(tmp, final)
                logger.checkpoint_saved(final, aggregator.update_count)

        except Exception as e:
            logger._log(f"⚠️  Error with {hospital_id}: {e}")
            break

    conn.close()
    logger.client_disconnected(hospital_id)


logger._log("\n🚀 ASYN-FEDPC Server Started")

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