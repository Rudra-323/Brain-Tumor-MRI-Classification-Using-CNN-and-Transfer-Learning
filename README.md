# 🧠 Brain Tumor MRI Classification using CNN

Classifies brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, or
**no tumor**, using transfer learning (VGG16), a fine-tuned classifier head, and
hyperparameter tuning with **Optuna**. Includes a **Streamlit** app for interactive
predictions.

> ⚠️ This is a student/capstone project, not a medical device. Predictions are not a
> diagnosis and should never be used for real clinical decisions.

---

## 📌 Overview

| | |
|---|---|
| **Task** | 4-class image classification |
| **Input** | Brain MRI scans (`.jpg`) |
| **Classes** | `glioma`, `meningioma`, `notumor`, `pituitary` |
| **Approach** | Architecture bake-off → fine-tune winner (VGG16) → Optuna tuning → final training → evaluation |
| **Best bake-off model** | VGG16 transfer learning (86.8% val. accuracy, frozen base) |
| **Deployment** | Streamlit web app (`app.py`) |

**Pipeline, in order:**
1. Load data from folders, split Training into Train/Validation (stratified 85/15).
2. EDA — class distribution, sample images.
3. Preprocessing — resize, rescale, augmentation, class weights.
4. **Bake-off** — 5 candidate architectures (2 custom CNNs + VGG16, MobileNetV2,
   ResNet50 transfer learning) trained briefly and compared → **VGG16 wins**.
5. **Fine-tune VGG16 first** — deeper classifier head (2-3 dense layers) + unfrozen top
   VGG16 layers, trained as a baseline.
6. **Optuna hyperparameter search** on that fine-tuned architecture (dense layer
   sizes/count, dropout, learning rate, optimizer, unfreeze depth).
7. Final model trained with the best hyperparameters found.
8. Evaluation on a fully held-out test set (accuracy, classification report, confusion
   matrix, ROC-AUC).

---

## 📂 Folder Structure

Suggested layout once everything is in the repo:

```
brain-tumor-cnn-classification/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── brain_tumor_cnn_classification.ipynb     # full training pipeline
│
├── app/
│   └── app.py                                   # Streamlit prediction app
│
├── models/
│   └── best_model_vgg16_transfer.keras           # saved after running the notebook (not committed — see below)
│
├── data/                                         # not committed — see Dataset section
│   └── brain-tumor-data/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── notumor/
│       │   └── pituitary/
│       └── Testing/
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
│
└── presentation/
    ├── brain_tumor_cnn_presentation.pptx
    └── presentation_script.txt
```

### What each file/folder is for

| Path | Purpose |
|---|---|
| `notebooks/brain_tumor_cnn_classification.ipynb` | End-to-end notebook: EDA → bake-off → fine-tuning → Optuna → final training → evaluation. |
| `app/app.py` | Streamlit app — upload an MRI image, get a predicted class + confidence. |
| `requirements.txt` | All Python dependencies for both the notebook and the app. |
| `models/` | Where the trained `.keras` model file lands after running the notebook. Large binary — keep out of git (see below). |
| `data/` | Dataset folder (Training/Testing). Not committed — download separately (see below). |
| `presentation/` | Slide deck + a plain-English narration script for presenting the project. |

---

## 🗂 Dataset

This project expects a 4-class brain tumor MRI dataset, already split into `Training/`
and `Testing/` folders, each containing one subfolder per class (`glioma`,
`meningioma`, `notumor`, `pituitary`) full of `.jpg` images.

1. Download the dataset (e.g. the "Brain Tumor MRI Dataset" on Kaggle — add your exact
   source link here).
2. Unzip it so it matches the `data/brain-tumor-data/` structure shown above.
3. In the notebook, point `TRAIN_DIR` / `TEST_DIR` at that folder.

The dataset is **not** committed to this repo (it's large, and redistribution may not
be permitted depending on the source's license) — download it yourself and place it
locally as above.

---

## ⚙️ Setup

**Requirements:** Python 3.9–3.11, pip. A GPU is optional but speeds up training a lot.

```bash
# 1. Clone the repo
git clone https://github.com/<Rudra-323>/brain-tumor-cnn-classification.git
cd brain-tumor-cnn-classification

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train the model (notebook)

```bash
jupyter notebook notebooks/brain_tumor_cnn_classification.ipynb
```

or open it in VS Code. Run all cells top to bottom. This will:
- Run the architecture bake-off
- Fine-tune VGG16 with a deeper classifier head
- Run the Optuna hyperparameter search
- Train the final model and save the best checkpoint to `models/best_model_vgg16_transfer.keras`
- Print/plot the full evaluation (accuracy, confusion matrix, ROC-AUC)

### 2. Run the Streamlit app

Make sure `best_model_vgg16_transfer.keras` (produced above) is in the same folder as
`app.py` (or update `MODEL_PATH` inside `app.py` to point to `models/`), then:

```bash
streamlit run app/app.py
```

Open the local URL Streamlit prints, upload an MRI image, and view the predicted class
and confidence.

---

## 📊 Results

| Architecture | Bake-off best val. accuracy |
|---|---|
| **VGG16 (transfer learning)** | **0.868** |
| MobileNetV2 (transfer learning) | 0.811 |
| Custom CNN v1 | 0.685 |
| ResNet50 (transfer learning) | 0.621 |
| Custom CNN v2 | 0.562 |
| Final Accuracy | **89.5 %** |

After fine-tuning VGG16 (deeper head + unfrozen layers) and tuning with Optuna:

| Metric | Value |
|---|---|
| Fine-tuned baseline val. accuracy | *fill in after running the notebook* |
| Optuna best val. accuracy | *fill in after running the notebook* |
| Final test accuracy | *fill in after running the notebook* |
| Final test loss | *fill in after running the notebook* |

---

## 🛠 Tech Stack

- **TensorFlow / Keras** — model building and training
- **VGG16** (ImageNet weights) — transfer learning backbone
- **Optuna** — hyperparameter tuning (TPE sampler + median pruning)
- **scikit-learn** — evaluation metrics
- **Streamlit** — web app for inference
- **pandas / matplotlib / seaborn** — data handling and visualization

---

## 🔮 Future Improvements

- Widen the Optuna search space (augmentation strength, batch size)
- Try additional backbones (EfficientNet, DenseNet121) in the bake-off
- Add Grad-CAM visualizations for model interpretability
- Use k-fold cross-validation for a more robust performance estimate

---

## 📄 License

Add a license of your choice (e.g. MIT) — create a `LICENSE` file in the repo root.

## 🙋 Author

*Rudrakanta Mandala* — feel free to contact : linkedin : [https://linkedin.com/in/rudrakanta].
