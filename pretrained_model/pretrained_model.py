import numpy as np
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
from keras.applications.resnet import ResNet50, preprocess_input

# Path to the Oxford-IIIT Pet image dataset.
images_path = "/Users/amiteshwarsingh/Documents/AI_ML_DL/classification_pet_faces/oxford_IIIT_pet_dataset/images"

def image_name_to_label(name):
    return ' '.join(os.path.splitext(name.lower())[0].rsplit('_')[:-1])

def assign_label_code(images_label):
    return {label : i for i,label in enumerate(images_label)}

def label_encode(images_label, label):
    code_dict = assign_label_code(images_label)
    label = label.lower()
    return code_dict.get(label, None)

# Collect filenames and derive the breed labels from the filename pattern.
images_name = [os.path.basename(file) for file in glob.glob(os.path.join(images_path, '*.jpg'))]

images_label = sorted(set(image_name_to_label(name) for name in images_name))
label_to_code = assign_label_code(images_label)

def features_and_labels(images_name):
    features = []
    labels = []
    IMG_SIZE = (224, 224)
    for name in images_name:
        label = image_name_to_label(name)
        label_code = label_to_code.get(label.lower())
        if label_code is None:
            continue
        # Load each image and resize it to the fixed model input size.
        img = load_img(os.path.join(images_path, name), target_size=(224, 224))
        img = img_to_array(img, dtype="uint8")
        features.append(img)
        labels.append(label_code)
    return np.array(features), np.array(labels)

features_array , labels_array = features_and_labels(images_name)

X_train, X_test, y_train, y_test = train_test_split(
    features_array,
    labels_array,
    test_size=0.2,
    random_state=42,
    stratify=labels_array
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

# Preprocess images the way ResNet50 expects.
X_train = preprocess_input(X_train.astype("float32"))
X_val = preprocess_input(X_val.astype("float32"))
X_test = preprocess_input(X_test.astype("float32"))

# Light augmentation helps the model generalize better.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(37, activation="softmax")
])

# Compile for the 37-breed classification problem.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

# Plot training curves for accuracy and loss.
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
plt.savefig(os.path.join(os.path.dirname(__file__), "pretrained_model_training_curves.png"))
plt.close()

# Final evaluation on the unseen test set.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss : {test_loss}, Test accuracy : {test_accuracy}")

# Save training, validation, and test results in one CSV.
history_df = pd.DataFrame(history.history)
history_df.insert(0, "phase", [f"epoch_{i + 1}" for i in range(len(history_df))])

test_df = pd.DataFrame([{
    "phase": "test",
    "loss": test_loss,
    "accuracy": test_accuracy,
    "val_loss": np.nan,
    "val_accuracy": np.nan,
}])

results_df = pd.concat([history_df, test_df], ignore_index=True)
results_df.to_csv(
    os.path.join(os.path.dirname(__file__), "pretrained_model_results.csv"),
    index=False,
)

