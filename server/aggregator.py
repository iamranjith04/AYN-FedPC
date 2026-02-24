import torch
import threading

class AsyncAggregator:
    def __init__(self, n_classes, alpha):
        self.n_classes = n_classes
        self.alpha = alpha
        self.lock = threading.Lock()
        self.global_protos = {}
        self.update_count = 0

    def update(self, local_protos):
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

            self.update_count += 1

    def get_global(self):
        with self.lock:
            return {
                cls: proto.clone().detach()
                for cls, proto in self.global_protos.items()
            }