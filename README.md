🩺 Skin Lesion Classification & Explainability System
📌 Overview

This project implements an end-to-end deep learning system for skin lesion classification using dermoscopic images.
The system is designed as a clinical decision-support tool and provides:

Automatic skin lesion classification

Model confidence score

Lesion localization using Grad-CAM

A user-friendly Streamlit interface for image upload and visualization

The project follows industry-standard machine learning practices with a clear separation between:

Backend (training & evaluation) → Jupyter Notebook

Frontend (inference & UI) → Streamlit application

🧠 Model Summary

Architecture: EfficientNet-B0

Pretraining: ImageNet

Classes:

BCC (Basal Cell Carcinoma)

BKL (Benign Keratosis)

MEL (Melanoma)

NV (Melanocytic Nevus)

Final Performance (Test Set)

Accuracy: 85.78%

Precision: 85.47%

Recall: 85.78%

F1-score: 85.50%

📁 Project Structure

Skin_infection/
├── Skin_Lesion_Classification.ipynb (Backend: training, evaluation, Grad-CAM)
├── app.py (Frontend: Streamlit UI)
├── best_model.pth (Trained model weights)
├── requirements.txt (Python dependencies)
└── README.md (Project documentation)

⚙️ Setup Instructions
1️⃣ Create and Activate Virtual Environment

Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

2️⃣ Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

🧪 Backend: Training & Evaluation (Jupyter Notebook)

File:
Skin_Lesion_Classification.ipynb

This notebook performs the following tasks:

Loads the ISIC 2019 dataset

Applies preprocessing and data augmentation

Trains EfficientNet-B0 using transfer learning

Evaluates the model on validation and test sets

Computes accuracy, precision, recall, and F1-score

Generates a confusion matrix

Implements Grad-CAM for explainability

Saves the best model as best_model.pth

To run the notebook:

jupyter notebook

Open Skin_Lesion_Classification.ipynb and run all cells from top to bottom.

Training is optional if best_model.pth is already available.

🖥️ Frontend: Streamlit User Interface

File:
app.py

Features:

Upload dermoscopic image

Display resized input image

Show predicted lesion class

Display confidence score

Visualize Grad-CAM heatmap

Display overlay on original image

To run the Streamlit app:

streamlit run app.py

Then open in browser:
http://localhost:8501

🔄 Inference Workflow

Upload a dermoscopic image

Image is preprocessed using the same pipeline as training

Model predicts lesion class and confidence

Grad-CAM highlights the lesion region

Results are displayed as:

Original image

Grad-CAM heatmap

Overlay visualization

🔍 Explainability (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to highlight image regions that most influence the model’s predictions.
This improves transparency and supports clinical interpretability.

Grad-CAM visualizations are qualitative and may vary slightly due to preprocessing and visualization scaling.

🧠 Design Decisions

EfficientNet-B0 chosen for strong performance with fewer parameters

Transfer learning used to improve convergence

Grad-CAM used instead of object detection to avoid bounding-box annotation requirements

Streamlit chosen for rapid and lightweight UI development

Clear separation between backend (.ipynb) and frontend (app.py)

📌 Final Note

All training, evaluation, and explainability logic is implemented in Skin_Lesion_Classification.ipynb.
The user-facing interface is implemented using Streamlit in app.py.

This design ensures modularity, maintainability, and ease of deployment.
