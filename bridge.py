from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import io, sys  

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

app = Flask(__name__)
CORS(app)

# ---- TEXT → SIGN ----

VIDEO_DIR = "F:\\asl\\dataset\\videos"
available_videos = {}

for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        file = f"{c}{ext}"
        path = os.path.join(VIDEO_DIR, file)
        if os.path.exists(path):
            available_videos[c] = file
            break


@app.route("/api/text-to-sign", methods=["POST"])
def text_to_sign():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"success":False,"error":"No text"}),400
    
    videos=[]
    for c in data["text"].upper():
        if c in available_videos:
            videos.append({"letter":c,"video_url":f"/videos/{available_videos[c]}"})

    return jsonify({"success":True,"videos":videos})


@app.route("/videos/<filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_DIR, filename)

# ---- SIGN → TEXT ----

MODEL_PATH="models/vgg16_asl.h5"
model=None

if os.path.exists(MODEL_PATH):
    model=load_model(MODEL_PATH)
else:
    print("Model missing!")

MAP_CHARACTERS={i:chr(i+65) for i in range(26)}  # 0=A ... 25=Z

@app.route("/api/sign-to-text", methods=["POST"])
def sign_to_text():
    if not model:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    files = request.files.getlist("video")  # 👈 get all uploaded files
    if len(files) == 0:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    result_text = ""

    try:
        for file in files:
            filename = file.filename.lower()
            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", file.filename)
            file.save(path)

            ext = filename.split(".")[-1]

            # ---- image processing ----
            if ext in ("jpg","jpeg","png","bmp","gif"):
                frame = cv2.imread(path)
                if frame is None:
                    raise Exception(f"Could not read image {filename}")
            else:
                # ---- video: read first frame ----
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise Exception(f"Could not read video {filename}")

            # ---- preprocess ----
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224,224)) / 255.0
            frame = np.expand_dims(frame, axis=0)

            # ---- predict ----
            pred = model.predict(frame, verbose=0)
            idx = int(np.argmax(pred))
            text = MAP_CHARACTERS.get(idx, "?")
            result_text += text  # append each prediction

            if os.path.exists(path):
                os.remove(path)

        return jsonify({
            "success": True,
            "text": result_text
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    finally:
        # clean up uploaded files
        for file in files:
            path = os.path.join("uploads", file.filename)
            if os.path.exists(path):
                os.remove(path)

    return jsonify({"success": True, "text": combined_text})



@app.route("/api/health")
def health():
    return jsonify({
        "status":"running",
        "model_loaded":model is not None,
        "available_videos":len(available_videos)
    })


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
