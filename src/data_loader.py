import glob
import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.utils import load_img, img_to_array


def image_name_to_label(name: str) -> str:
    return " ".join(os.path.splitext(name.lower())[0].rsplit("_")[:-1])


def assign_label_code(images_label):
    return {label: i for i, label in enumerate(images_label)}


def load_image_names(images_path):
    return [Path(file).name for file in glob.glob(str(Path(images_path) / "*.jpg"))]


def features_and_labels(images_name, images_path, label_to_code, img_size=(224, 224)):
    features = []
    labels = []
    for name in images_name:
        label = image_name_to_label(name)
        label_code = label_to_code.get(label.lower())
        if label_code is None:
            continue
        img = load_img(Path(images_path) / name, target_size=img_size)
        img = img_to_array(img, dtype="uint8")
        features.append(img)
        labels.append(label_code)
    return np.array(features), np.array(labels)


def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, label_to_code):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "y_test.npy", y_test)

    with open(output_dir / "label_to_code.json", "w", encoding="utf-8") as file:
        json.dump(label_to_code, file, indent=2)


def load_processed_data(output_dir):
    output_dir = Path(output_dir)
    X_train = np.load(output_dir / "X_train.npy")
    X_val = np.load(output_dir / "X_val.npy")
    X_test = np.load(output_dir / "X_test.npy")
    y_train = np.load(output_dir / "y_train.npy")
    y_val = np.load(output_dir / "y_val.npy")
    y_test = np.load(output_dir / "y_test.npy")

    with open(output_dir / "label_to_code.json", "r", encoding="utf-8") as file:
        label_to_code = json.load(file)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_to_code


def load_data(raw_images_path, img_size=(224, 224), test_size=0.2, val_size=0.25, random_state_test=42, random_state_val=1):
    images_name = load_image_names(raw_images_path)
    images_label = sorted(set(image_name_to_label(name) for name in images_name))
    label_to_code = assign_label_code(images_label)
    features_array, labels_array = features_and_labels(images_name, raw_images_path, label_to_code, img_size=img_size)

    X_train, X_test, y_train, y_test = train_test_split(
        features_array,
        labels_array,
        test_size=test_size,
        random_state=random_state_test,
        stratify=labels_array,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state_val,
        stratify=y_train,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_to_code": label_to_code,
        "num_classes": len(label_to_code),
    }
