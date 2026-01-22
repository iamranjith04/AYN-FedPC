The script split_bloodmnist_npz.py prepares the BloodMNIST dataset for Federated Learning experiments. It downloads the source data and partitions it into "siloed" hospital datasets to simulate a Non-IID (Non-Independent and Identically Distributed) data distribution.

To simulate a realistic decentralized environment, the data is split into three virtual hospitals. Each hospital only has access to a specific subset of the 8 total classes, with some intentional overlaps:
        "hospital_1": [0, 1, 2, 3],
        "hospital_2": [3, 4, 5, 6],
        "hospital_3": [2, 5, 6, 7]