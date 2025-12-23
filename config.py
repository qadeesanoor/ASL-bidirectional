# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

IMAGE_SIZE = 50
VGG_IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 30
EPOCHS = 10
RANDOM_STATE = 42

TRAIN_DIR = r"F:\asl\dataset\asl_alphabet_train"

LABEL_MAP = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
    'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28
}

MAP_CHARACTERS = {v: k for k, v in LABEL_MAP.items()}
