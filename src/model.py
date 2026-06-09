import tensorflow as tf
from keras.applications.resnet import ResNet50

from .features import build_augmentation


def build_basic_model(num_classes=37, input_shape=(224, 224, 3)):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            build_augmentation(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def build_pretrained_model(num_classes=37, input_shape=(224, 224, 3)):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            build_augmentation(),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MyModel:
    def __init__(self, model_type="basic", num_classes=37, input_shape=(224, 224, 3)):
        self.model_type = model_type
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self._build()

    def _build(self):
        if self.model_type == "basic":
            return build_basic_model(num_classes=self.num_classes, input_shape=self.input_shape)
        if self.model_type == "pretrained":
            return build_pretrained_model(num_classes=self.num_classes, input_shape=self.input_shape)
        raise ValueError("model_type must be 'basic' or 'pretrained'")

    def compile(self):
        return compile_model(self.model)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def summary(self):
        return self.model.summary()

    def save(self, *args, **kwargs):
        return self.model.save(*args, **kwargs)
