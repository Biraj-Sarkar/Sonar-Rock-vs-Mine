# Sonar Rock vs Mine Prediction 🛥️⚓

## 📌 Project Overview
This project implements a **machine learning–based classification system** to predict whether an underwater object detected by sonar signals is a **Rock** or a **Mine**.  
The model is trained on numerical sonar signal features and evaluated using multiple classification algorithms.

The project demonstrates a **complete end-to-end ML workflow**, from data preprocessing to model comparison and batch prediction.

---

## 🧠 Problem Statement
Sonar systems emit sound waves underwater and analyze the reflected signals to detect objects.  
However, distinguishing **mines** from **rocks** is challenging due to similar sonar reflections.

**Objective:**  
Build a machine learning model that accurately classifies sonar signals into:
- **Rock (R)**
- **Mine (M)**

This is a **binary classification problem**.

---

## 📊 Dataset Description
- Dataset: Sonar Dataset (UCI Machine Learning Repository / Kaggle)
- Total Samples: 208
- Features: 60 continuous numerical values  
  - Represent energy of sonar signals at different frequency bands
- Target Variable:
  - `R` → Rock
  - `M` → Mine

Each row corresponds to one sonar observation.

---

## 🛠️ Technologies Used
- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 🔄 Project Workflow

1. **Data Loading & Inspection**
   - Load dataset from CSV
   - Verify structure, shape, and cleanliness

2. **Exploratory Data Analysis (EDA)**
   - Analyze class distribution
   - Inspect statistical properties of features

3. **Data Preprocessing**
   - Encode target labels (`R → 0`, `M → 1`)
   - Train–test split (80% / 20%) with stratification
   - Feature scaling using `StandardScaler`

4. **Model Training**
   - Baseline model: Logistic Regression

5. **Model Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report

6. **Model Comparison**
   Compared the following models using identical preprocessing:
   - Logistic Regression
   - Support Vector Classifier (SVC)
   - Decision Tree Classifier
   - Random Forest Classifier

7. **Prediction System**
   - Batch prediction using CSV input
   - Outputs predictions and confidence scores to a new CSV file

---

## 📈 Model Comparison Summary

| Model | Description |
|------|------------|
| Logistic Regression | Simple linear baseline |
| SVC | Strong performance on high-dimensional data |
| Decision Tree | Interpretable but prone to overfitting |
| Random Forest | More robust ensemble of trees |

Evaluation was done using **test accuracy** under the same conditions for all models.

---

## 📂 Prediction System (CSV-Based)
The project supports batch prediction using unseen sonar data stored in a CSV file.

**Input:**  
- CSV file with 60 feature columns (no labels)

**Output:**  
- CSV file with:
  - Predicted class (Rock / Mine)
  - Probability of Mine

This simulates a real-world ML inference pipeline.

---

## 🚀 How to Run the Project

Clone the repository:
   ```bash
   git clone https://github.com/Biraj-Sarkar/Sonar-Rock-vs-Mine.git
   ```
