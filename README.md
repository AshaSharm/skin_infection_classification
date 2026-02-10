# Skin Lesion Classification & Explainability System

## Overview
This project implements an **end-to-end deep learning system** for skin lesion classification using dermoscopic images.  
The system is designed as a **clinical decision-support tool** and provides:

- Automatic skin lesion classification
- Model confidence score
- Lesion localization using **Grad-CAM**
- A **user-friendly Streamlit interface** for image upload and visualization

The project follows industry-standard machine learning practices with a clear separation between:
- **Backend (training & evaluation)** → Jupyter Notebook  
- **Frontend (inference & UI)** → Streamlit application  

---

## Model Summary
- **Architecture:** EfficientNet-B0  
- **Pretraining:** ImageNet  
- **Classes:**  
  - BCC (Basal Cell Carcinoma)  
  - BKL (Benign Keratosis)  
  - MEL (Melanoma)  
  - NV (Melanocytic Nevus)  

### Final Performance (Test Set)
- **Accuracy:** 85.78%  
- **Precision:** 85.47%  
- **Recall:** 85.78%  
- **F1-score:** 85.50%  

---

##  Project Structure



Skin_infection/
│
├── Skin_Lesion_Classification.ipynb # Backend: training, evaluation, Grad-CAM
├── app.py # Frontend: Streamlit UI
├── best_model.pth # Trained model weights
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

##  Setup Instructions

### 1️⃣ Create and Activate Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

2️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

Backend: Training & Evaluation (Jupyter Notebook)
File
Skin_Lesion_Classification.ipynb

What this notebook does:

Loads the ISIC 2019 dataset

Applies preprocessing and data augmentation

Trains EfficientNet-B0 using transfer learning

Evaluates model on validation and test sets

Computes accuracy, precision, recall, F1-score

Generates confusion matrix

Implements Grad-CAM for explainability

Saves the best model as best_model.pth

How to run:
jupyter notebook


Then open:

Skin_Lesion_Classification.ipynb


Run all cells top to bottom.

⚠️ Training is optional if best_model.pth is already available.

Frontend: Streamlit User Interface
File
app.py

Features:

Upload dermoscopic image

Display resized input image

Show predicted lesion class

Display confidence score

Visualize:

Grad-CAM heatmap

Overlay of heatmap on original image

Run the Streamlit App
streamlit run app.py


Then open the displayed URL in your browser:

http://localhost:8501

Inference Workflow (Frontend)

Upload a dermoscopic image

Image is preprocessed using the same pipeline as training

Model predicts lesion class and confidence

Grad-CAM highlights lesion region

Results are visualized side-by-side:

Original image

Grad-CAM heatmap

Overlay visualization

Explainability (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to:

Highlight image regions influencing model decisions

Improve transparency and clinical interpretability

Design Decisions

EfficientNet-B0 chosen for strong performance with fewer parameters

Transfer learning used to improve convergence

Grad-CAM selected instead of object detection to avoid bounding-box annotation requirements

Streamlit used for rapid, lightweight UI development

Clear separation of backend (.ipynb) and frontend (app.py)