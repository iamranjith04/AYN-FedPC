# ASYC - FEDPC #

ASYN-FEDPC is a federated learning system where multiple hospitals train local models on private patient data and share only compressed prototype vectors — never raw data. A central server aggregates these prototypes asynchronously and returns updated global prototypes that guide future local training.

### How it works: ###

Each hospital runs the full training loop independently. The flow per round is:
        
        • Step 1 — Load data: NPZDataset loads images and labels from .npz files. Each hospital only sees its own patient data partition.
        
        • Step 2 — Forward pass: The CNN produces both a classification output (out) and a 64-dim feature vector (feat) from the encoder.
        
        • Step 3 — Compute loss: Two losses are combined: cross-entropy loss on local labels, and prototype regularization loss against global prototypes.
        
        • Step 4 — Backprop & update: opt.step() updates all model weights — encoder + classifier head.
        
        • Step 5 — Build prototypes: After all epochs, build_prototypes() runs a forward pass over the full dataset and averages features per class.
        
        • Step 6 — Send to server: The local prototype dict {class_id: tensor} is pickled and sent to the server.
        
        • Step 7 — Receive global: The server responds with updated global prototypes. These are used in the next round's proto loss.

**The training loss is:**

    loss = CE_loss + lambda * proto_loss

    where:

        CE_loss    = CrossEntropy(model_output, local_labels)
    
        proto_loss = mean MSE(feat[i], global_proto[y_i])   for all i where y_i exists in global_protos
    
        lambda     = LAMBDA_PROTO = 2.5   (from config)

### How to Run: ###

**To run Server machine:** 

    python -m server.server 

**To run Hospitals Individual machine:**

    python -m hospital.client hospital_1
    python -m hospital.client hospital_2
    python -m hospital.client hospital_3


**To evaluate Cross validation after Training:** 

    python cross_hospital_eval.py --model local_models/hospital_1_round_9.pt --source fedpc_bloodmnist_npz/hospital_1/train.npz --target fedpc_bloodmnist_npz/hospital_3/val.npz