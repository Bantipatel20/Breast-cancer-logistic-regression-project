# Breast Cancer Classification Using Logistic Regression

## Overview
This project implements a logistic regression model to classify breast cancer diagnoses (Malignant or Benign) using the Breast Cancer Wisconsin dataset. The model is trained using gradient descent, and the cost function convergence is visualized to evaluate the training process. The dataset is preprocessed, split into training, validation, and test sets, and the model's performance is evaluated using accuracy metrics.

## Dataset
The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset (`breast-cancer.csv`). It contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. Each row represents a sample with the following columns:

- **id**: Unique identifier for each sample.
- **diagnosis**: Target variable (M for Malignant, B for Benign).
- **Features**: 30 numerical features describing characteristics of the cell nuclei, such as:
  - `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.
  - Standard error and "worst" values for each feature (e.g., `radius_se`, `radius_worst`).

## Data Preprocessing
1. The `diagnosis` column is mapped to binary values: **M → 1 (Malignant), B → 0 (Benign)**.
2. Features are normalized using **z-score normalization**:
   ```
   x' = (x - mean) / std
   ```
3. The dataset is split into **training (60%)**, **validation (20%)**, and **test (20%)** sets using `train_test_split` with a `random_state` of 60 for reproducibility.

## Model Implementation
The logistic regression model is implemented from scratch using the following steps:

1. **Sigmoid Function**:
   ```
   σ(z) = 1 / (1 + exp(-z))
   ```
   where `z = w^T X + b`.

2. **Cost Function (Log Loss)**:
   ```
   J(w, b) = -(1/m) * Σ [y * log(ŷ) + (1 - y) * log(1 - ŷ)]
   ```

3. **Gradient Descent for Parameter Updates**:
   - Weight update:
     ```
     w := w - α * ∂J/∂w
     ```
   - Bias update:
     ```
     b := b - α * ∂J/∂b
     ```

4. **Model Evaluation**:
   - **Accuracy**:
     ```
     Accuracy = (correct predictions / total samples) * 100
     ```
   - **Confusion Matrix**
   - **Precision, Recall, F1-score**

## Visualization
- Cost function convergence is plotted to evaluate model training.
- Decision boundary visualization (if applicable).
- Performance metrics such as ROC curve and AUC score.

## Recommendations
- Fix the cost function implementation to use the correct log-loss formula.
- Experiment with a smaller learning rate (e.g., **α = 0.01** or **α = 0.001**).
- Double-check feature normalization to ensure all features are properly scaled.
- Monitor gradients to detect if they are exploding or vanishing.

## Dependencies
Install the required dependencies using:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Running the Model
Run the script using:
```bash
python breast_cancer_logistic_regression.py
```

## Conclusion
This project demonstrates how logistic regression can be used for binary classification of breast cancer diagnosis using numerical features extracted from FNA images. The model performance can be improved by optimizing hyperparameters, enhancing feature engineering, and experimenting with different learning rates.
