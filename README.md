# Food Image Classification with Calorie Estimation

This project implements deep learning models for classifying food images (11 classes from Food-101) and estimating calories.

## features
- **Custom CNN**: A lightweight custom Convolutional Neural Network.
- **High Accuracy Model**: Transfer Learning using EfficientNet-B0 (>84% accuracy).
- **Streamlit App**: Interactive web interface for predictions.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Custom CNN Model (Lightweight)
- **Training**:
  ```bash
  python train_food11.py
  ```
- **Run App**:
  ```bash
  python -m streamlit run app.py
  ```

### 2. High Accuracy Model (EfficientNet)
Located in `high_acc_model/` folder.
- **Training**:
  ```bash
  cd high_acc_model
  python train.py
  ```
- **Run App**:
  ```bash
  cd high_acc_model
  python -m streamlit run app.py
  ```

## Project Structure
- `app.py`: Streamlit app for Custom CNN.
- `train_food11.py`: Training script for Custom CNN.
- `food_model.pth`: Trained Custom CNN model.
- `high_acc_model/`: Folder containing High Accuracy model scripts.
    - `train.py`: Training script (EfficientNet).
    - `app.py`: Streamlit app (EfficientNet).
    - `effective_food_model.pth`: Trained EfficientNet model.
- `data/`: Dataset directory (excluded from git).
- `requirements.txt`: Python dependencies.
