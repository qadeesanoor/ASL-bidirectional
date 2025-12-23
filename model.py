# models.py
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from config import NUM_CLASSES, VGG_IMAGE_SIZE, MODEL_DIR

def build_vgg16():
    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(VGG_IMAGE_SIZE, VGG_IMAGE_SIZE, 3)
    )

    for layer in base.layers:
        layer.trainable = False

    x = Flatten()(base.output)
    out = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_model(model):
    model.save(os.path.join(MODEL_DIR, "vgg16_asl.keras"))
    with open(os.path.join(MODEL_DIR, "vgg16_asl.json"), "w") as f:
        f.write(model.to_json())
