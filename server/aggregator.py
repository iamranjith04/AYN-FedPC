import torch
import threading

class AsyncAggregator:
    def __init__(self, n_classes, alpha):
        self.n_classes = n_classes
        self.alpha = alpha
        self.lock = threading.Lock()
        self.global_protos = {}
        self.update_count = 0

        # NEW: store the latest prototype submission per hospital
        # This is what cluster_prototypes() needs — one proto dict per hospital
        self.hospital_protos = {}

    def update(self, hospital_id, local_protos):
        with self.lock:
            for cls, proto in local_protos.items():
                proto = proto.clone().detach()

                if cls not in self.global_protos:
                    self.global_protos[cls] = proto
                else:
                    self.global_protos[cls] = (
                        self.alpha * self.global_protos[cls] +
                        (1 - self.alpha) * proto
                    )

            # NEW: overwrite with latest submission for this hospital
            self.hospital_protos[hospital_id] = {
                cls: proto.clone().detach()
                for cls, proto in local_protos.items()
            }

            self.update_count += 1

    def get_global(self):
        with self.lock:
            return {
                cls: proto.clone().detach()
                for cls, proto in self.global_protos.items()
            }

    # NEW: return snapshot of all hospital protos for clustering
    def get_all_hospital_protos(self):
        with self.lock:
            return {
                h: {cls: p.clone().detach() for cls, p in protos.items()}
                for h, protos in self.hospital_protos.items()
            }

    # NEW: how many distinct hospitals have submitted at least once
    def n_hospitals(self):
        with self.lock:
            return len(self.hospital_protos)