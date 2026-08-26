# Smart Parking Security System

> **Face & License Plate Verification for Parking Access**

## Overview

This project is a computer vision-based parking security system designed to verify both the **driver's face** and the **vehicle license plate** during vehicle entry and exit.

The system uses a dual-verification approach so that a vehicle is only allowed to exit when the detected license plate is registered and the driver's facial features are sufficiently similar to the data captured during entry.

The project was developed as a final academic project and includes both the implementation code and a complete project report.

## Main Features

- Face detection using **Haar Cascade**
- Face feature extraction using **Histogram of Oriented Gradients (HOG)**
- Face matching using **Cosine Similarity**
- License plate region detection using image processing and contour analysis
- License plate recognition using **Tesseract OCR / Pytesseract**
- Automatic validation between entry and exit data
- Dual-camera image capture for face and license plate
- Entry and exit record management
- Parking record search by license plate
- Local data persistence using Pickle
- Automatic saving of preprocessed face and plate images

## System Concept

The system records two primary pieces of information when a vehicle enters the parking area:

1. Driver's face
2. Vehicle license plate

When the vehicle attempts to leave, the system captures both again and compares them with the stored entry data.

```text
VEHICLE ENTRY
     |
     +--> Capture Driver Face
     |        |
     |        +--> Face Detection
     |        +--> HOG Feature Extraction
     |
     +--> Capture License Plate
              |
              +--> Plate Detection
              +--> Image Preprocessing
              +--> OCR Recognition
                      |
                      v
                Save Entry Data
                      |
                      v
                PARKING DATABASE

VEHICLE EXIT
     |
     +--> Capture Face + License Plate
                      |
                      v
            Compare with Entry Data
                      |
             +--------+--------+
             |                 |
          MATCHED           NOT MATCHED
             |                 |
        Exit Allowed        Exit Denied
```

## Computer Vision Methods

### 1. Face Detection

The system uses OpenCV's **Haar Cascade Classifier** to locate the driver's face in an image.

Detected faces are cropped and resized before further processing.

### 2. HOG Feature Extraction

**Histogram of Oriented Gradients (HOG)** is used to represent facial characteristics as a feature vector.

The image is:

- resized,
- converted to grayscale,
- histogram-equalized,
- processed using image gradients,
- divided into cells and blocks,
- normalized into a HOG descriptor.

### 3. Face Verification

Face feature vectors captured during entry and exit are compared using **Cosine Similarity**.

In the current implementation, the vehicle is allowed to exit when:

```text
Face Similarity > 0.70
```

and the license plate matches an existing entry record.

### 4. License Plate Detection

The license plate region is detected using image-processing steps including:

- grayscale conversion,
- Gaussian blur,
- adaptive thresholding,
- contour detection,
- aspect-ratio filtering,
- area filtering,
- region cropping.

The current implementation searches for candidate regions with an aspect ratio between approximately **1.5 and 6** and an area above **5000 pixels**.

### 5. License Plate Recognition

The detected plate region is preprocessed and passed to **Pytesseract**, a Python interface for Tesseract OCR.

Several OCR page segmentation modes are tested, while recognized characters are limited to:

```text
A-Z
0-9
```

The result is then processed using a regular-expression pattern to obtain a structured license plate string.

## Parking Workflow

### Vehicle Entry

When `Vehicle Entry` is selected:

1. Two cameras are initialized.
2. The system performs a countdown before capturing images.
3. The driver's face is captured.
4. The face is detected and processed.
5. HOG features are extracted.
6. The license plate image is captured.
7. The plate region is detected and preprocessed.
8. OCR converts the plate characters into text.
9. The license plate and face feature data are stored as an entry record.

### Vehicle Exit

When `Vehicle Exit` is selected:

1. The face and license plate are captured again.
2. OCR reads the current license plate.
3. The system searches for the plate in the entry records.
4. HOG features from the current face are compared with the stored face data.
5. Cosine similarity is calculated.
6. If the plate exists and facial similarity passes the threshold, exit is approved.
7. The corresponding entry is removed from the active parking records.
8. If verification fails, the vehicle is denied exit.

### Search Records

The application also provides a search feature that allows the operator to:

- search for a specific license plate, or
- display currently registered parking entries.

## Application Menu

The terminal interface provides four options:

```text
=== Parking System Menu ===
1. Vehicle Entry
2. Vehicle Exit
3. Search Records
4. Exit Program
========================
```

## Technologies

### Programming Language

- Python

### Main Libraries

- OpenCV
- NumPy
- Pytesseract
- Scikit-learn
- Pillow
- TensorFlow / Keras
- Pickle
- Threading

> Some imported libraries are retained from the development code even if the current implementation does not use every imported module directly.

## Hardware

The project uses two camera inputs:

- Camera 1 for capturing the driver's face
- Camera 2 for capturing the vehicle license plate

In the implementation:

```python
cv2.VideoCapture(0)
cv2.VideoCapture(1)
```

The project report describes testing with a **1080p webcam** and a **720p laptop camera**.

## Installation

### 1. Install Python

Python 3.x is recommended.

### 2. Install Required Packages

```bash
pip install opencv-python numpy pytesseract scikit-learn pillow tensorflow
```

### 3. Install Tesseract OCR

Tesseract OCR must also be installed separately.

On Windows, the source code currently points to:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is installed in another directory, update the following line in the Python file:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 4. Connect Two Cameras

Make sure both camera devices are detected by the computer.

If the camera indexes are different, adjust:

```python
cv2.VideoCapture(0)
cv2.VideoCapture(1)
```

## Running the Program

Run the main Python file:

```bash
python parking_system.py
```

The terminal will display the parking system menu.

## Data Storage

Parking records are stored locally inside:

```text
parking_data/
```

The program uses:

```text
entries.pkl
```

to retain active parking entry information.

The system also creates directories for processed face and license plate data.

Example structure:

```text
parking_data/
├── entries.pkl
├── faces/
└── plates/
```

Preprocessed images are also saved with timestamps during the entry and exit process.

## Project Structure

A recommended GitHub repository structure is:

```text
Smart-Parking-Security-System/
│
├── parking_system.py
├── README.md
├── LAPORAN PROJECT AKHIR UAS_KELOMPOK G.pdf
│
└── parking_data/
    ├── faces/
    └── plates/
```

> `parking_data/` is generated by the application and does not need to contain active parking records when the project is first uploaded.

## Project Report

A complete PDF report documenting the development, methodology, testing, results, and discussion of this project is included in this repository:

**`LAPORAN PROJECT AKHIR UAS_KELOMPOK G.pdf`**

The report covers topics including:

- background and motivation for parking security,
- related computer vision methods,
- Haar Cascade face detection,
- HOG feature extraction,
- cosine similarity,
- license plate OCR,
- system workflow,
- image preprocessing,
- entry and exit simulations,
- testing results,
- system limitations,
- conclusions and future improvements.

For a more detailed explanation of the system and its experimental results, please refer to the PDF report.

## Experimental Notes

During testing, the system was able to verify matching face and license plate data under suitable conditions.

One documented successful exit test produced a **face similarity value of 0.84** together with a matching license plate.

The project also showed that detection performance is affected by:

- lighting conditions,
- camera angle,
- image quality,
- distance between the subject and camera.

For this reason, consistent lighting and suitable camera positioning are important for reliable operation.

## Limitations

The current prototype has several limitations:

- Detection accuracy can decrease under poor or changing lighting.
- License plate OCR depends strongly on image clarity.
- Camera angle can affect face and plate detection.
- The system is designed as a prototype rather than a production parking access system.
- The current implementation relies on local data storage.
- Face verification uses handcrafted HOG features rather than a modern deep-learning face embedding model.

## Future Development

Possible improvements include:

- deep-learning-based face recognition,
- more robust automatic license plate recognition,
- low-light image enhancement,
- database integration,
- graphical user interface,
- real-time parking dashboard,
- automatic gate control,
- cloud-based record storage,
- notification system for rejected exit attempts.

## Purpose

This project demonstrates how **computer vision and dual identity verification** can be integrated into a parking security system to reduce reliance on manual verification and provide an additional security layer for vehicle entry and exit.

---

### Smart Parking Security System

**Face + License Plate = Dual Verification**
