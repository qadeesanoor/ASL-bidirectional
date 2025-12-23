# dataset_utils.py
import tensorflow as tf
from config import VGG_IMAGE_SIZE, BATCH_SIZE

def create_dataset(X, y, batch_size=BATCH_SIZE, shuffle=True):
    """
    Creates a tf.data.Dataset from NumPy arrays for efficient training.

    Parameters:
    - X: numpy array of images
    - y: numpy array of one-hot labels
    - batch_size: int, batch size
    - shuffle: bool, whether to shuffle the dataset

    Returns:
    - tf.data.Dataset
    """

    # Convert numpy arrays to tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), seed=42)

    # Resize images on the fly (useful if using VGG16 or other pretrained models)
    dataset = dataset.map(
        lambda img, label: (tf.image.resize(img, (VGG_IMAGE_SIZE, VGG_IMAGE_SIZE)), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
