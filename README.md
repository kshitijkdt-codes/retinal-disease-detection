# Retinal Disease Detection

A Streamlit web application that uses a trained TensorFlow/Keras image-classification model to display scores for retinal fundus image classes.

> **Medical disclaimer:** This is an educational/student project, not a medical device. It has not been established as safe or clinically validated for diagnosis. Do not use its output to make healthcare decisions.

## Features

- Upload JPG, JPEG, or PNG images.
- Convert images to RGB and resize them to 224 × 224 pixels.
- Apply TensorFlow's ResNet50 `preprocess_input` function.
- Display the model's score for each configured class and the highest-scoring class.
- Show useful error messages if the model is missing or its output shape does not match the labels.

## Classes configured in the app

The label order currently follows the project report:

1. Normal
2. Diabetic Retinopathy
3. Glaucoma
4. Cataract
5. AMD
6. Hypertension
7. Myopia
8. Other

**Important:** The labels and their order must exactly match the classes used when training the model. Verify this with your original training code before using the app.

## Requirements

- Python 3.10 or 3.11 is a practical starting point for TensorFlow compatibility.
- A trained Keras model file named `retinal_disease_model.h5`.

The model weights are **not included** in this repository because they were not supplied with the report. The app will show a message until you put the model file beside `app.py`. The model must accept 224 × 224 RGB images prepared with ResNet50 preprocessing and return eight class scores in the order above.

## Run locally

### 1. Download or clone this repository

If you downloaded the ZIP, extract it and open the extracted folder in a terminal.

If you use Git:

```bash
git clone YOUR_REPOSITORY_URL
cd retinal-disease-detection
```

Replace `YOUR_REPOSITORY_URL` with your own GitHub repository URL.

### 2. Create a virtual environment (recommended)

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the trained model

Copy your trained `retinal_disease_model.h5` file into this folder, next to `app.py`.

Do not download or use a random model from the internet as a substitute. If you no longer have the model, you will need the original model/weights or retrain it using a properly labeled dataset.

### 5. Start the app

```bash
streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## Model and evaluation notes

- The application code does not train a model; it loads an already trained model.
- The class names above are taken from the project report. The actual training class mapping was not included in the report's application code, so it must be verified.
- The score is shown as a percentage for readability. It should not be interpreted as a clinically calibrated probability.
- A roughly square aspect ratio cannot reliably prove that an image is a retinal fundus image. This app validates that the upload is a readable image, but it cannot guarantee the image is a fundus photograph.
- The report mentions an approximately two-second response time. That figure has not been independently benchmarked in this repository.

## Suggested repository structure

```text
retinal-disease-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── retinal_disease_model.h5   # local only; ignored by Git
```

## Future improvements

- Verify the exact model output activation and training class order.
- Add a separate training/evaluation notebook with dataset provenance and reproducible splits.
- Report evaluation metrics on a held-out test set (for example, per-class precision, recall, F1-score, and confusion matrix).
- Add tests for image preprocessing and model output validation.
- Consider model hosting or Git LFS only after checking file size, licensing, privacy, and platform limits.
