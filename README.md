# ASL Bidirectional Translator

## American Sign Language ↔ Text Translation System

An AI-powered **bidirectional American Sign Language (ASL) communication system** that translates **ASL signs into text** and **text into corresponding ASL signs**. The project aims to reduce communication barriers between sign language users and non-signers through machine learning and computer vision.

---

## Overview

The project provides two-way communication between ASL and text:

* **Sign-to-Text:** Recognizes ASL signs from images and converts them into readable text.
* **Text-to-Sign:** Takes text as input and provides the corresponding ASL sign representation.

The combination of both directions creates a **bidirectional communication system** for ASL users and non-signers.

---

## Key Features

* **ASL Sign Recognition**
* **Sign-to-Text Translation**
* **Text-to-Sign Conversion**
* **Single Image Testing**
* **Multiple Image Testing**
* **Machine Learning-Based Prediction**
* **Image Processing**
* **Frontend Interface**
* **Visualization Utilities**

---

## System Workflow

| Direction        | Input     | Processing                          | Output   |
| ---------------- | --------- | ----------------------------------- | -------- |
| **Sign-to-Text** | ASL Image | Image Processing → Model Prediction | Text     |
| **Text-to-Sign** | Text      | Text Processing → Sign Mapping      | ASL Sign |

### Sign-to-Text

```text
ASL Image
    ↓
Image Preprocessing
    ↓
Feature Extraction
    ↓
Trained Model
    ↓
Sign Prediction
    ↓
Text Output
```

### Text-to-Sign

```text
Text Input
    ↓
Text Processing
    ↓
Sign Mapping
    ↓
ASL Representation
    ↓
Sign Output
```

---

## Project Structure

| File               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `train.py`         | Trains the machine learning model                    |
| `model.py`         | Contains the model architecture and prediction logic |
| `data_loader.py`   | Loads and prepares the dataset                       |
| `dataset_utils.py` | Provides dataset processing utilities                |
| `config.py`        | Contains project configuration                       |
| `test1img.py`      | Tests the model using a single image                 |
| `testmulimg.py`    | Tests the model using multiple images                |
| `text2sign.py`     | Handles Text-to-Sign functionality                   |
| `frontend.html`    | Provides the frontend interface                      |
| `bridge.py`        | Handles communication between components             |
| `callbackutlis.py` | Contains callback-related utilities                  |
| `visualization.py` | Provides visualization functionality                 |

---

## Technologies Used

| Technology           | Purpose                               |
| -------------------- | ------------------------------------- |
| **Python**           | Core development and machine learning |
| **Machine Learning** | ASL sign classification               |
| **Computer Vision**  | Processing and analyzing sign images  |
| **HTML**             | Frontend interface                    |
| **Image Processing** | Preparing input images for prediction |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/qadeesanoor/ASL-bidirectional.git
```

### Navigate to the Project Directory

```bash
cd ASL-bidirectional
```

### Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model

Run the training script:

```bash
python train.py
```

### Test a Single Image

Use the single-image testing script:

```bash
python test1img.py
```

### Test Multiple Images

Use the multiple-image testing script:

```bash
python testmulimg.py
```

### Text-to-Sign

Run the Text-to-Sign component:

```bash
python text2sign.py
```

---

## Project Goals

The main objectives of this project are:

* Develop an automated **ASL recognition system**.
* Convert ASL gestures into understandable text.
* Convert text into corresponding ASL signs.
* Reduce communication barriers between ASL users and non-signers.
* Explore the application of **machine learning and computer vision** in accessibility.
* Build a foundation for real-time bidirectional sign language communication.

---

## Future Improvements

* **Real-time ASL recognition** using a webcam.
* Support for **continuous ASL sentences**.
* Expand the supported ASL vocabulary.
* Improve model accuracy and robustness.
* Add **Text-to-Speech** functionality.
* Develop a fully deployed web application.
* Add support for mobile platforms.
* Enable real-time bidirectional communication.

---
