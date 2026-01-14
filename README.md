# Sonar Rock vs Mine Prediction 🛥️⚓

A complete **end-to-end Machine Learning project** that classifies underwater objects as **Rock** or **Mine** using sonar signal data, and is deployed as an **interactive Streamlit web application**.


## 📌 Project Overview
Sonar systems are widely used to detect underwater objects, but distinguishing **mines** from **rocks** is challenging due to similar sonar reflections.  
This project applies **machine learning classification techniques** to solve this problem using numerical sonar signal features.

The project covers:
- Data analysis and preprocessing
- Model training and evaluation
- Model comparison
- Batch prediction using CSV files
- Deployment using **Streamlit**

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
- **Streamlit**

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

8. **Deployment**
   - Deployed as a Streamlit web application
   - Users can upload CSV files and download predictions

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

## 🌐 Streamlit Web Application

### Features
- Upload a CSV file containing **60 sonar features**
- Automatic validation of input format
- Predicts **Rock / Mine** for each sample
- Displays **Mine probability (confidence)**
- Download predictions as a CSV file

### Input Format
- CSV file with **exactly 60 columns**
- No target/label column required

### Output
- Prediction (`Rock` / `Mine`)
- Probability of Mine for each sample

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Biraj-Sarkar/Sonar-Rock-vs-Mine.git
cd Sonar-Rock-vs-Mine
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📂 Project Structure
sonar-rock-vs-mine/
│
├── app.py                    # Streamlit application
├── sonar_model.pkl           # Trained ML model
├── scaler.pkl                # Feature scaler
├── requirements.txt
├── README.md
└── sonar_rock_vs_mine.ipynb  # Development notebook

---

## ✅ Key Learnings
* Built a complete ML pipeline from data to deployment
* Understood model behavior through comparison
* Implemented batch prediction using real-world CSV input
* Deployed an ML model as an interactive web application
* Debugged real-world Python and ML issues

---

## 🔮 Future Enhancements
* Hyperparameter tuning and cross-validation
* Model selection inside the Streamlit app
* Single-sample manual input UI
* Cloud deployment with persistent storage
* Authentication and logging

---

## 👤 Author

Biraj Sarkar  
B.Tech in Computer Science and Engineering  
IIT Guwahati  

## 📜 License

This project is intended for educational and academic purposes.
