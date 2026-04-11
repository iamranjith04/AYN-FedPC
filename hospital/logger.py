import os
import torch
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _log(filepath, line):
    """Append a line to a log file and also print it."""
    print(line)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _proto_detail_lines(protos, indent="      "):
    """
    For each class in protos, return lines showing:
      - first 8 weight values  (the actual tensor values)
      - min / max / mean / std / L2-norm
    """
    lines = []
    for cls in sorted(protos.keys()):
        vec = protos[cls].float()                       # ensure float32 for stats
        first8 = vec[:8].tolist()
        first8_str = "  ".join(f"{v:+.4f}" for v in first8)

        lines.append(
            f"{indent}class {cls} │ "
            f"norm={vec.norm().item():.4f}  "
            f"mean={vec.mean().item():+.4f}  "
            f"std={vec.std().item():.4f}  "
            f"min={vec.min().item():+.4f}  "
            f"max={vec.max().item():+.4f}"
        )
        lines.append(
            f"{indent}         weights[0:8] → [ {first8_str} ]"
        )
    return lines


# ─────────────────────────────────────────────
#  CLIENT-SIDE LOGGING
# ─────────────────────────────────────────────

class ClientLogger:
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id
        self.path = os.path.join(LOG_DIR, f"{hospital_id}.log")
        # clear log at start of new run
        open(self.path, "w", encoding="utf-8").close()
        self._log(f"{'='*70}")
        self._log(f"  HOSPITAL LOG — {hospital_id}   started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*70}")

    def _log(self, line):
        _log(self.path, line)

    def round_start(self, r, total_rounds, lam):
        self._log(f"\n{'─'*70}")
        self._log(f"  ROUND {r+1}/{total_rounds}   [{_ts()}]   λ_proto = {lam:.2f}")
        self._log(f"{'─'*70}")

    def train_result(self, acc):
        self._log(f"  [TRAIN]  Local accuracy = {acc:.2f}%")

    def log_sending_protos(self, local_protos):
        self._log(f"\n  [SEND]  📡 Sending {len(local_protos)} prototypes to server  [{_ts()}]")
        self._log(f"  {'─'*60}")
        for line in _proto_detail_lines(local_protos):
            self._log(line)
        self._log(f"  {'─'*60}")

    def log_received_protos(self, global_protos):
        self._log(f"\n  [RECV]  📥 Received {len(global_protos)} prototypes from server  [{_ts()}]")
        self._log(f"  {'─'*60}")
        for line in _proto_detail_lines(global_protos):
            self._log(line)
        self._log(f"  {'─'*60}")

    def log_proto_delta(self, local_protos, global_protos):
        """
        Log cosine similarity between what was sent (local) and what was
        received (global/cluster). This tells you how much the server moved
        each class prototype relative to your local version.
        """
        shared = set(local_protos.keys()) & set(global_protos.keys())
        if not shared:
            return
        self._log(f"\n  [DELTA] Cosine similarity: local_sent vs received")
        for cls in sorted(shared):
            lv = local_protos[cls].float().unsqueeze(0)
            gv = global_protos[cls].float().unsqueeze(0)
            sim = torch.nn.functional.cosine_similarity(lv, gv, dim=1).item()
            drift = (lv - gv).norm().item()
            self._log(
                f"      class {cls} │ cosine_sim={sim:.4f}   L2_drift={drift:.4f}"
            )

    def model_saved(self, path):
        self._log(f"  [SAVE]  💾 Model saved → {path}")

    def training_complete(self):
        self._log(f"\n{'='*70}")
        self._log(f"  {self.hospital_id} — TRAINING COMPLETE   [{_ts()}]")
        self._log(f"{'='*70}\n")


# ─────────────────────────────────────────────
#  SERVER-SIDE LOGGING
# ─────────────────────────────────────────────

class ServerLogger:
    def __init__(self):
        self.path = os.path.join(LOG_DIR, "server.log")
        open(self.path, "w", encoding="utf-8").close()
        self._log(f"{'='*70}")
        self._log(f"  SERVER LOG   started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"{'='*70}")

    def _log(self, line):
        _log(self.path, line)

    def client_connected(self, hospital_id, addr):
        self._log(f"\n  [CONNECT]  🔗 {hospital_id}  from {addr}   [{_ts()}]")

    def client_disconnected(self, hospital_id):
        self._log(f"  [DISCONNECT]  ❌ {hospital_id}   [{_ts()}]")

    def log_received_protos(self, hospital_id, local_protos, update_count):
        self._log(f"\n  [RECV]  📦 {hospital_id} — update #{update_count}   [{_ts()}]")
        self._log(f"  Received {len(local_protos)} class prototypes:")
        self._log(f"  {'─'*60}")
        for line in _proto_detail_lines(local_protos):
            self._log(line)
        self._log(f"  {'─'*60}")

    def log_sending_protos(self, hospital_id, response_protos, cluster_id=None):
        label = f"cluster {cluster_id}" if cluster_id is not None else "global (fallback)"
        self._log(f"\n  [SEND]  📡 → {hospital_id}   source: {label}   [{_ts()}]")
        self._log(f"  Sending {len(response_protos)} class prototypes:")
        self._log(f"  {'─'*60}")
        for line in _proto_detail_lines(response_protos):
            self._log(line)
        self._log(f"  {'─'*60}")

    def log_global_state(self, global_protos):
        self._log(f"\n  [GLOBAL]  Current global proto state   [{_ts()}]")
        self._log(f"  {'─'*60}")
        for line in _proto_detail_lines(global_protos):
            self._log(line)
        self._log(f"  {'─'*60}")

    def checkpoint_saved(self, path, update_count):
        self._log(f"  [CKPT]  💾 Checkpoint saved → {path}   update #{update_count}")

    def clustering_result(self, hospital_to_cluster):
        self._log(f"\n  [CLUSTER]  K-Means result   [{_ts()}]")
        for h, c in sorted(hospital_to_cluster.items()):
            self._log(f"      {h} → cluster {c}")