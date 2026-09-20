# Brain Tumor MRI Classification — Project Workflow

## 1. Business Need
Reading a brain MRI scan and spotting a tumor takes an experienced radiologist and
real time. In busy hospitals or under-resourced clinics, that time isn't always
available, and delays in flagging a serious scan can cost precious time in
treatment. A fast, automated first-pass check can help prioritize which scans a
doctor reviews first.

## 2. Problem Statement
Build a system that looks at a brain MRI scan and automatically classifies it into
one of four categories:
- **Glioma**
- **Meningioma**
- **Pituitary tumor**
- **No tumor**

This is a supportive screening tool, not a diagnostic replacement — it never
replaces a radiologist's judgment.

## 3. Objective
Train a deep learning image classifier that is both **accurate** and **reliable**,
by testing multiple approaches fairly instead of committing to the first idea that
works, then packaging the result as something people can actually use.

## 4. Dataset
- Brain MRI images, already sorted into `Training/` and `Testing/` folders, each
  split into the four class folders above.
- Training data further split 85% / 15% into Train and Validation (stratified by
  class).
- Testing folder is kept fully separate and only used once, at the very end.

## 5. Workflow

| Step | What happens | Why |
|---|---|---|
| **1. Explore the data** | Check class balance, view sample scans per class | Catch labeling or balance issues early |
| **2. Preprocess** | Resize to 150×150, rescale pixels, augment training images, compute class weights | Make training stable and fair across classes |
| **3. Architecture bake-off** | Train 5 candidate models (2 custom CNNs + VGG16, MobileNetV2, ResNet50) briefly under identical conditions | Pick the best starting architecture with evidence, not guesswork — **VGG16 won** (86.8% val. accuracy) |
| **4. Fine-tune the winner** | Add a deeper classifier head (2–3 dense layers) to VGG16 and unfreeze its top layers | Adapt VGG16's general image knowledge specifically to MRI scans |
| **5. Hyperparameter tuning (Optuna)** | Automatically search learning rate, dropout, dense layer sizes, optimizer, and unfreeze depth | Squeeze out better performance than hand-picked settings, without guesswork |
| **6. Final training** | Train the tuned model fully, saving only the best checkpoint | Avoid overfitting; keep the best version, not just the last one |
| **7. Evaluation** | Test once on the untouched Testing set — accuracy, confusion matrix, ROC-AUC | Get an honest, unbiased measure of real-world performance |
| **8. Deployment** | Wrap the final model in a Streamlit app | Let anyone upload a scan and get an instant prediction |

## 6. Tech Stack
TensorFlow / Keras · VGG16 (transfer learning) · Optuna · scikit-learn · Streamlit ·
pandas / matplotlib / seaborn

## 7. Outcome
A trained classifier that reliably sorts brain MRI scans into the four categories,
delivered as a simple web app — going from raw folders of images to something a
non-technical person can actually use in a browser.

## 8. Future Scope
- Try newer architectures (EfficientNet, DenseNet121) in the bake-off
- Add Grad-CAM so the model can visually show *why* it made a prediction
- Use k-fold cross-validation for a more robust performance estimate
- Widen the Optuna search (augmentation strength, batch size)
