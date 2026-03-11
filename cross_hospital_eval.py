import torch
from torch.utils.data import DataLoader
import argparse

from hospital.model import CNN
from hospital.dataset import NPZDataset


def evaluate(model_path, data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model:", model_path)
    print("Evaluating on dataset:", data_path)

    # Load dataset
    dataset = NPZDataset(data_path)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Load model
    model = CNN(len(dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, yl, yg in loader:
            x = x.to(device)
            yl = yl.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == yl).sum().item()
            total += yl.size(0)

    acc = 100 * correct / total
    print("\nCross-hospital evaluation result")
    print("---------------------------------")
    print(f"Samples evaluated : {total}")
    print(f"Accuracy          : {acc:.2f}%")

    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved hospital model"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Validation dataset of another hospital"
    )

    args = parser.parse_args()

    evaluate(args.model, args.data)

"""
python cross_hospital_eval.py \
--model local_models/hospital_1_update_10.pt \
--data fedpc_bloodmnist_npz/hospital_2/val.npz

"""