import numpy as np 
import glob 
import os 
import tensorflow as tf 
from keras.utils import load_img, img_to_array
import pandas as pd 
from sklearn.model_selection import train_test_split

# path to image dataset folder 
images_path = "/Users/amiteshwarsingh/Documents/AI_ML_DL/classification_pet_faces/oxford-iiit-pet/images"

def image_name_to_label(name):
    return ' '.join(os.path.splitext(name.lower())[0].rsplit('_')[:-1])

def assign_label_code(images_label):
    return {label : i for i,label in enumerate(images_label)}

def label_encode(images_label, label):
    code_dict = assign_label_code(images_label)
    label = label.lower()
    return code_dict.get(label, None)

# names retrival from folder and lable code assignment  
images_name = [os.path.basename(file) for file in glob.glob(os.path.join(images_path,'*.jpg'))]

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
        img = load_img(os.path.join(images_path, name), target_size=IMG_SIZE)
        img = img_to_array(img, dtype="uint8")
        features.append(img)
        labels.append(label_code)
    return np.array(features), np.array(labels)

features_array , labels_array = features_and_labels(images_name)

# splitting data in training and test 
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
    random_state=1,
    stratify=y_train
)

# normalizing trainging data images 
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# augmenting data to vary data for generalization
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), # 50 % neurons gets turned off during training 
    tf.keras.layers.Dense(37, activation='softmax')
])

# compiling the model 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# actual evaluation on test dataset 
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss : {test_loss}, Test accuracy : {test_accuracy}")

# store training, validation, and test results in one CSV
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
    os.path.join(os.path.dirname(__file__), "training_validation_test_results.csv"),
    index=False,
)
