import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def cluster_prototypes(all_prototypes, n_clusters):
    hospital_ids = list(all_prototypes.keys())
    
    if len(hospital_ids) < n_clusters:
        n_clusters = len(hospital_ids)
    
    hospital_vectors = []
    for hospital in hospital_ids:
        protos = all_prototypes[hospital]
        if len(protos) > 0:
            all_class_protos = torch.stack([p for p in protos.values()])
            avg_proto = all_class_protos.mean(0)
            hospital_vectors.append(avg_proto.numpy())
        else:
            hospital_vectors.append(np.zeros(64))
    
    hospital_vectors = np.array(hospital_vectors)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(hospital_vectors)
    
    hospital_to_cluster = {
        hospital: int(cluster_labels[i])
        for i, hospital in enumerate(hospital_ids)
    }
    
    cluster_prototypes = defaultdict(dict)
    for hospital, cluster_id in hospital_to_cluster.items():
        for class_id, proto in all_prototypes[hospital].items():
            if class_id not in cluster_prototypes[cluster_id]:
                cluster_prototypes[cluster_id][class_id] = []
            cluster_prototypes[cluster_id][class_id].append(proto)
    
    global_protos = {}
    for class_id in range(8):
        class_protos = []
        for hospital_protos in all_prototypes.values():
            if class_id in hospital_protos:
                class_protos.append(hospital_protos[class_id])
        
        if class_protos:
            global_protos[class_id] = torch.stack(class_protos).mean(0)
    
    cluster_class_protos = {}
    for cluster_id, class_dict in cluster_prototypes.items():
        cluster_class_protos[cluster_id] = {}
        for class_id, protos in class_dict.items():
            cluster_class_protos[cluster_id][class_id] = torch.stack(protos).mean(0)
    
    return global_protos, hospital_to_cluster, cluster_class_protos