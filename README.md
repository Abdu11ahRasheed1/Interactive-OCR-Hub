# Interactive OCR Hub

An interactive Web Application for Optical Character Recognition (OCR) converting binary-grid characters based on Hu Moments + K-Nearest Neighbors (K-NN) and Keras CNN classifiers. Developed on top of Python, FastAPI, and Vanilla HTML/JS.

## Features

* **Segment OCR Grid Patterns**: Dynamically processes images containing character grids (e.g. grids of 'a', 'd', 's', 'o', etc.), segments character components, and catalogs them ready for classifier training.
* **7-Dimensional Hu Moments Extraction**: Extracts features for segmented characters matching structural shape descriptors.
* **Euclidean K-NN Classifier**: Uses scikit-learn standard scaling and `KNeighborsClassifier` to perform character recognitions.
* **Keras CNN Classifier**: Asynchronously fits a Conv2D classifier model on character labels directory. Includes fallback support if TensorFlow is not installed.
* **Interactive Predicted Grid Canvas**: Renders test files on Canvas and paints dynamically shaded prediction boxes, showing hovering stats and Hu moments details inside.

## Directory Structure

```
├── backend/
│   ├── app.py            # FastAPI main router API
│   ├── ocr_engine.py     # Segmentation, Feature extraction, Model training and prediction
│   ├── models/           # Stores scaler.pkl, knn_model.pkl, and cnn_model.keras
│   └── dataset/          # Grid template datasets segmented folders
├── frontend/
│   ├── index.html        # Glassmorphic UI Dashboard page
│   ├── css/
│   │   └── style.css     # Dark mode CSS library
│   └── js/
│       └── app.js        # Canvas drawing and upload listeners
├── requirements.txt      # Project requirements
└── README.md             # Guide documentation
```

## Getting Started

### Prerequisites

Identify that Python 3.8+ is installed:
```bash
python --version
```

### Installation

1. Clone or copy files into the local workspace directory.
2. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   ```
   python -m pip install -r requirements.txt
   ```
   *(Note: if TensorFlow/Keras is not installed locally, the backend will scale back to the K-NN classifier seamlessly without training/testing crashes).*

### How to Run

1. Run the FastAPI backend:
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```
2. Open your web browser and navigate to:
   [http://localhost:8000](http://localhost:8000)

## Dataset & Training Guide
1. Launch the web dashboard.
2. In the **Model Studio**, upload character grid templates (e.g. `a.bmp`, `d.bmp` grids) selecting class identifier. The backend will automatically segment grid items and construct directories.
3. Click on the **Train Models** button. The engine will run training fits.
4. Go to **OCR Playground**, upload test images (e.g. `test1.bmp`), choose method (KNN or CNN), and view recognition predictions.
