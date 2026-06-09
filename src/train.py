from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

from .data_loader import (
    assign_label_code,
    features_and_labels,
    image_name_to_label,
    load_image_names,
    load_processed_data,
    save_processed_data,
)
from .features import preprocess_basic, preprocess_pretrained
from .model import build_basic_model, build_pretrained_model, compile_model


def split_data(features_array, labels_array, test_size=0.2, val_size=0.25, random_state_test=42, random_state_val=1):
    """Split data into train, validation, and test sets using stratified sampling."""
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
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_training(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """Fit the model with early stopping on validation loss."""
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    return history


def save_results(history, test_loss, test_accuracy, output_path):
    """Persist per-epoch training history plus final test metrics to a CSV file."""
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "phase", [f"epoch_{i + 1}" for i in range(len(history_df))])
    test_df = pd.DataFrame(
        [
            {
                "phase": "test",
                "loss": test_loss,
                "accuracy": test_accuracy,
                "val_loss": np.nan,
                "val_accuracy": np.nan,
            }
        ]
    )
    results_df = pd.concat([history_df, test_df], ignore_index=True)
    results_df.to_csv(output_path, index=False)
    return results_df


def load_config(project_root):
    """Load hyperparameters and paths from config.yaml at the project root."""
    config_path = Path(project_root) / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment(model_name="basic"):
    """End-to-end training pipeline: load/preprocess data, train, evaluate, save."""
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root)

    images_path = project_root / config["data"]["images_dir"]
    outputs_dir = project_root / config["paths"]["outputs_dir"] / f"{model_name}_model"
    processed_dir = project_root / config["data"]["processed_dir"]

    outputs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_files = [
        processed_dir / "X_train.npy",
        processed_dir / "X_val.npy",
        processed_dir / "X_test.npy",
        processed_dir / "y_train.npy",
        processed_dir / "y_val.npy",
        processed_dir / "y_test.npy",
        processed_dir / "label_to_code.json",
    ]

    if all(path.exists() for path in processed_files):
        X_train, X_val, X_test, y_train, y_val, y_test, label_to_code = load_processed_data(processed_dir)
    else:
        images_name = load_image_names(images_path)
        images_label = sorted(set(image_name_to_label(name) for name in images_name))
        label_to_code = assign_label_code(images_label)
        features_array, labels_array = features_and_labels(
            images_name,
            images_path,
            label_to_code,
            tuple(config["training"]["img_size"]),
        )

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            features_array,
            labels_array,
            test_size=config["training"]["test_size"],
            val_size=config["training"]["val_size"],
            random_state_test=config["training"]["random_state_test"],
            random_state_val=config["training"]["random_state_val"],
        )

        save_processed_data(processed_dir, X_train, X_val, X_test, y_train, y_val, y_test, label_to_code)

    num_classes = len(label_to_code)

    if model_name == "basic":
        X_train = preprocess_basic(X_train)
        X_val = preprocess_basic(X_val)
        X_test = preprocess_basic(X_test)
        model = build_basic_model(num_classes=num_classes, input_shape=tuple(config["training"]["img_size"]) + (3,))
        epochs = config["training"]["epochs_basic"]
    elif model_name == "pretrained":
        X_train = preprocess_pretrained(X_train)
        X_val = preprocess_pretrained(X_val)
        X_test = preprocess_pretrained(X_test)
        model = build_pretrained_model(num_classes=num_classes, input_shape=tuple(config["training"]["img_size"]) + (3,))
        epochs = config["training"]["epochs_pretrained"]
    else:
        raise ValueError("model_name must be 'basic' or 'pretrained'")

    compile_model(model)
    history = run_training(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=config["training"]["batch_size"],
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    save_results(history, test_loss, test_accuracy, outputs_dir / f"{model_name}_results.csv")
    model.save(outputs_dir / f"{model_name}_model.keras")
    return model, history, (test_loss, test_accuracy)


def main():
    parser = ArgumentParser(description="Train the pet face classification models")
    parser.add_argument(
        "--model",
        choices=["basic", "pretrained"],
        default="basic",
        help="Which model to train (default: basic)",
    )
    args = parser.parse_args()
    _, _, metrics = run_experiment(args.model)
    print(f"Finished training {args.model} model. Test metrics: {metrics}")


if __name__ == "__main__":
    main()
