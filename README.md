# Brain Tumor MRI Classification (ResNet50, PyTorch & Streamlit)

This repository contains a **multiclass brain tumor MRI classifier** using **ResNet50 (PyTorch)** and a **Streamlit demo app** for predictions.

The model classifies MRI images into four categories:
- **glioma**
- **meningioma**
- **pituitary**
- **no_tumor**

---

## Project Structure

```
.
├── dataset/
│   ├── Testing/
│   └── Training/
├── models/
│   ├── model.pt
│   ├── class_names.json
│   └── ...
├── main.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Usage

### Train Model (Jupyter Notebook)

Use the provided notebook (`.ipynb`) to train the model on the brain tumor MRI dataset. This includes:
1. Loading images with `ImageFolder`
2. Transfer learning with ResNet50
3. Evaluation + Confusion Matrix
4. Saving artifacts (`model.pt`, `class_names.json`)

### Run Streamlit Demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo app:

```
streamlit run streamlit_app.py
```

Upload a brain MRI image and see the predicted class along with confidence scores.

## Dataset

This project uses the Brain Tumor MRI Dataset from Kaggle:

🔗: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

Please download and extract the dataset locally before running the training notebook.
Organize folders as:

```
Training/
  glioma/
  meningioma/
  pituitary/
  no_tumor/
Testing/
  glioma/
  meningioma/
  pituitary/
  no_tumor/
```

## Credits

Thanks to:

- Masoud Nickparvar for publishing the brain MRI dataset on Kaggle.
- The PyTorch and Streamlit open source communities.

## Disclaimer

This classifier is for educational and demonstration purposes only, submission for 888351 | Modern Computer Vision And Applications For Entrepreneur.
It is not intended for clinical use or medical diagnosis.
