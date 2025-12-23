# callbacks_utils.py
import numpy as np
import os
from tensorflow.keras.callbacks import Callback
from config import LOG_DIR

class MetricsCheckpoint(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(os.path.join(LOG_DIR, "training_logs.npy"), self.history)
