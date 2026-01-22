import os
import numpy as np
import medmnist
from medmnist import INFO

DATASET_NAME = "bloodmnist"
SAVE_DIR = "fedpc_bloodmnist_npz"

hospital_classes = {
    "hospital_1": [0, 1, 2, 3],
    "hospital_2": [3, 4, 5, 6],
    "hospital_3": [2, 5, 6, 7]
}

def save_npz(images, labels, split_name):
    for hospital, class_ids in hospital_classes.items():
        idx = np.isin(labels, class_ids)

        imgs_h = images[idx]
        labels_h = labels[idx]

        save_path = os.path.join(SAVE_DIR, hospital)
        os.makedirs(save_path, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_path, f"{split_name}.npz"),
            images=imgs_h,
            labels=labels_h
        )


def main():
    info = INFO[DATASET_NAME]
    DataClass = getattr(medmnist, info['python_class'])

    train_data = DataClass(split='train', download=True)
    val_data   = DataClass(split='val', download=True)
    test_data  = DataClass(split='test', download=True)

    y_train = train_data.labels.squeeze()
    y_val   = val_data.labels.squeeze()
    y_test  = test_data.labels.squeeze()

    save_npz(train_data.imgs, y_train, "train")
    save_npz(val_data.imgs, y_val, "val")
    save_npz(test_data.imgs, y_test, "test")

    print("BloodMNIST saved as NPZ files (no image conversion).")


if __name__ == "__main__":
    main()
