import os
import numpy as np
import medmnist
from medmnist import INFO
from sklearn.model_selection import StratifiedShuffleSplit

DATASET_NAME = "bloodmnist"
SAVE_DIR = "fedpc_bloodmnist_npz"
HOSPITALS = ["hospital_1", "hospital_2", "hospital_3"]
SPLIT_RATIOS = [0.33, 0.33, 0.34]


HOSPITAL_MISSING_CLASSES = {
    "hospital_1": [],        # sees all classes
    "hospital_2": [2, 5],    # misses class 2 and 5
    "hospital_3": []        # misses class 4
}

os.makedirs(SAVE_DIR, exist_ok=True)

def save_split(images, labels, split_name):
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - SPLIT_RATIOS[0],
        random_state=42
    )

    idx1, rest = next(sss.split(images, labels))

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        random_state=42
    )

    idx2, idx3 = next(sss2.split(images[rest], labels[rest]))
    idx2, idx3 = rest[idx2], rest[idx3]

    splits = {
        "hospital_1": idx1,
        "hospital_2": idx2,
        "hospital_3": idx3
    }

    for h, idx in splits.items():
        imgs = images[idx]
        labs = labels[idx]

        # -------- NEW: REMOVE MISSING CLASSES --------
        missing = HOSPITAL_MISSING_CLASSES[h]
        if len(missing) > 0:
            keep = ~np.isin(labs, missing)
            imgs = imgs[keep]
            labs = labs[keep]

        path = os.path.join(SAVE_DIR, h)
        os.makedirs(path, exist_ok=True)

        np.savez_compressed(
            os.path.join(path, f"{split_name}.npz"),
            images=imgs,
            labels=labs
        )

        print(
            f"{h} | {split_name} | "
            f"classes={sorted(set(labs.tolist()))}"
        )

def main():
    info = INFO[DATASET_NAME]
    DataClass = getattr(medmnist, info["python_class"])

    train = DataClass(split="train", download=True)
    val   = DataClass(split="val", download=True)
    test  = DataClass(split="test", download=True)

    save_split(train.imgs, train.labels.squeeze(), "train")
    save_split(val.imgs,   val.labels.squeeze(),   "val")
    save_split(test.imgs,  test.labels.squeeze(),  "test")

    print("\n✅ Sample-based split with controlled class missingness completed.")

if __name__ == "__main__":
    main()
