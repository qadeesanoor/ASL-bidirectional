# data_loader.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from config import IMAGE_SIZE, LABEL_MAP, DATA_DIR

def get_data(folder):
    X, y = [], []

    for cls in os.listdir(folder):
        if cls not in LABEL_MAP:
            continue

        label = LABEL_MAP[cls]
        cls_path = os.path.join(folder, cls)

        for img_name in tqdm(os.listdir(cls_path)):
            img = cv2.imread(os.path.join(cls_path, img_name))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize(img, (IMAGE_SIZE, IMAGE_SIZE, 3), preserve_range=True)
            X.append(img.astype(np.float32) / 255.0)
            y.append(label)

    return np.array(X, np.float32), np.array(y, np.int32)

def save_numpy(X_train, y_train, X_test, y_test):
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

def load_numpy():
    return (
        np.load(os.path.join(DATA_DIR, "X_train.npy")),
        np.load(os.path.join(DATA_DIR, "y_train.npy")),
        np.load(os.path.join(DATA_DIR, "X_test.npy")),
        np.load(os.path.join(DATA_DIR, "y_test.npy"))
    )
