# train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from config import *
from data_loader import get_data, save_numpy, load_numpy
from dataset_utils import create_dataset
from model import build_vgg16
from callbackutlis import MetricsCheckpoint
from visualization import save_learning_curves

import sys


# Ensure console uses UTF-8
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8')


# ======================================================
# 1️⃣ Load Data (Use local .npy if exists)
# ======================================================
np_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in np_files):
    print("✔ Loading data from local .npy files")
    X_train, y_train_hot, X_test, y_test_hot = load_numpy()
else:
    print("⏳ Loading images and creating .npy files")
    X, y = get_data(TRAIN_DIR)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    y_train_hot = to_categorical(y_train, NUM_CLASSES)
    y_test_hot = to_categorical(y_test, NUM_CLASSES)
    save_numpy(X_train, y_train_hot, X_test, y_test_hot)

# ======================================================
# 2️⃣ Create tf.data Datasets
# ======================================================
train_ds = create_dataset(X_train, y_train_hot, BATCH_SIZE)
test_ds = create_dataset(X_test, y_test_hot, BATCH_SIZE, shuffle=False)

# ======================================================
# 3️⃣ Compute Class Weights
# ======================================================
y_train_classes = np.argmax(y_train_hot, axis=1)
cw = dict(enumerate(
    class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_classes),
        y=y_train_classes
    )
))

# ======================================================
# 4️⃣ Define Save Function
# ======================================================
def save_model_all(model):
    # Full model
    model.save(os.path.join(MODEL_DIR, "vgg16_asl.keras"))
    # Architecture only
    with open(os.path.join(MODEL_DIR, "vgg16_asl.json"), "w") as f:
        f.write(model.to_json())
    # Weights only
    model.save_weights(os.path.join(MODEL_DIR, "vgg16_asl.weights.h5"))
    print("✔ Model saved to /models")

# ======================================================
# 5️⃣ Build and Train Model
# ======================================================
model = build_vgg16()
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    class_weight=cw,
    callbacks=[MetricsCheckpoint()],
    verbose=1
)

# ======================================================
# 6️⃣ Save Trained Model & Learning Curves
# ======================================================
save_model_all(model)
save_learning_curves(history)

# ======================================================
# 7️⃣ Evaluate
# ======================================================
y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true = np.argmax(y_test_hot, axis=1)

print("✔ Classification Report:")
print(classification_report(y_true, y_pred, target_names=MAP_CHARACTERS.values()))
