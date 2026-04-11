import torch
from torch.utils.data import DataLoader
import argparse

from hospital.model import CNN
from hospital.dataset import NPZDataset
from hospital.prototype import build_prototypes


def load_encoder(model, model_path, device):
    """
    Load only encoder weights (ignore classifier head)
    """
    state_dict = torch.load(model_path, map_location=device)

    # Remove classifier (fc) weights
    state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}

    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate(model_path, train_data_path, test_data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🔍 Cross-Hospital Evaluation")
    print("----------------------------------")
    print(f"Model from      : {model_path}")
    print(f"Prototype built : {train_data_path}")
    print(f"Test dataset    : {test_data_path}")
    print("----------------------------------")

    # ✅ Load dataset for prototype building (source hospital)
    train_dataset = NPZDataset(train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    # ✅ Load dataset for evaluation (target hospital)
    test_dataset = NPZDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # ✅ Build model
    model = CNN(num_classes=1)  # classifier not used
    model = load_encoder(model, model_path, device)
    model.to(device)
    model.eval()

    # ✅ Build prototypes from source hospital
    print("\n📦 Building source prototypes...")
    global_protos = build_prototypes(model, train_loader)

    for cls, proto in global_protos.items():
        print(f"   class {cls} -> norm {proto.norm().item():.4f}")

    # ✅ Evaluate using prototype similarity
    print("\n🧪 Evaluating on target hospital...")

    correct = 0
    total = 0

    with torch.no_grad():
        for x, _, yg in test_loader:
            x = x.to(device)
            yg = yg.to(device)

            _, feats = model(x, return_feat=True)

            preds = []
            for f in feats:
                sims = {}

                for cls, proto in global_protos.items():
                    sims[cls] = torch.cosine_similarity(
                        f, proto.to(device), dim=0
                    )

                pred = max(sims, key=sims.get)
                preds.append(pred)

            preds = torch.tensor(preds, device=device)

            correct += (preds == yg).sum().item()
            total += yg.size(0)

    acc = 100.0 * correct / total

    print("\n📊 RESULT")
    print("----------------------------------")
    print(f"Samples evaluated : {total}")
    print(f"Accuracy          : {acc:.2f}%")
    print("----------------------------------")

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
        "--source",
        type=str,
        required=True,
        help="Source hospital TRAIN dataset (for prototypes)"
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target hospital TEST/VAL dataset"
    )

    args = parser.parse_args()

    evaluate(args.model, args.source, args.target)