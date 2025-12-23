# visualization.py
import os
import matplotlib.pyplot as plt
from config import OUTPUT_DIR

def save_learning_curves(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['train','val'])
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train','val'])
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
