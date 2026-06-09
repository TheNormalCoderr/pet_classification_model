"""
Runner script for the ResNet50 transfer learning experiment.

Usage (from project root):
    python scripts/run_pretrained_model.py

Outputs are saved to: outputs/pretrained_model/
  - pretrained_model_training_curves.png
  - pretrained_model_results.csv
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import assign_label_code, features_and_labels, load_image_names, image_name_to_label
from src.features import preprocess_pretrained
from src.model import build_pretrained_model, compile_model
from src.train import run_training, save_results, split_data

IMAGES_PATH = PROJECT_ROOT / "data" / "raw" / "oxford_IIIT_pet_dataset" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pretrained_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

images_name = load_image_names(IMAGES_PATH)
images_label = sorted(set(image_name_to_label(name) for name in images_name))
label_to_code = assign_label_code(images_label)
features_array, labels_array = features_and_labels(images_name, IMAGES_PATH, label_to_code)

X_train, X_val, X_test, y_train, y_val, y_test = split_data(features_array, labels_array)
X_train = preprocess_pretrained(X_train)
X_val = preprocess_pretrained(X_val)
X_test = preprocess_pretrained(X_test)

model = compile_model(build_pretrained_model(num_classes=len(images_label)))
history = run_training(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)

epochs_range = range(1, len(history.history["accuracy"]) + 1)
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history["accuracy"], label="Training accuracy")
plt.plot(epochs_range, history.history["val_accuracy"], label="Validation accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history["loss"], label="Training loss")
plt.plot(epochs_range, history.history["val_loss"], label="Validation loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pretrained_model_training_curves.png")
plt.close()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

results_df = save_results(
    history,
    test_loss,
    test_accuracy,
    OUTPUT_DIR / "pretrained_model_results.csv",
)
print(results_df.tail(1))
