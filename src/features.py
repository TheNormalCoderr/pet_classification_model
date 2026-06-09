import tensorflow as tf
from keras.applications.resnet import preprocess_input


def build_augmentation():
    """Build a data augmentation pipeline for on-the-fly augmentation during training."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )


def preprocess_basic(x):
    """Normalize pixel values to [0, 1] for the custom CNN model."""
    return x.astype("float32") / 255.0


def preprocess_pretrained(x):
    """Apply ResNet50 ImageNet preprocessing (channel-wise mean subtraction)."""
    return preprocess_input(x.astype("float32"))
