import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model(r"F:\asl\models\vgg16_asl.h5")

# Class map
map_characters1 = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',
    12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',
    22:'W',23:'X',24:'Y',25:'Z',26:'del',27:'nothing',28:'space',29:'other'
}

# list of 4 images
image_paths = [
    r"F:\asl\dataset\asl_alphabet_test\H_test.jpg",
    r"F:\asl\dataset\asl_alphabet_test\e_test.jpg",
    r"F:\asl\dataset\asl_alphabet_test\l_test.jpg",
    r"F:\asl\dataset\asl_alphabet_test\l_test.jpg",
     r"F:\asl\dataset\asl_alphabet_test\o_test.jpg"
]

predicted_sequence = ""  # store final string

for img_path in image_paths:
    img = cv2.imread(img_path)

    if img is None:
        print("Could not read:", img_path)
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    pred_probs = model.predict(img_resized)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    pred_label = map_characters1[pred_class]

    predicted_sequence += pred_label

print("You said:", predicted_sequence)
