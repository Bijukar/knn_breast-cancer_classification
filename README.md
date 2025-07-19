
# 🧠 KNN Breast Cancer Classification

This project implements a **K-Nearest Neighbors (KNN)** classifier on the **Breast Cancer  dataset** from kaggle dataset. The model is trained, tuned, evaluated, and saved for future use.
This repository contains code for classifying breast cancer as either malignant or benign using the K-Nearest Neighbors (KNN) algorithm. The project follows a standard machine learning workflow, including data loading, cleaning, exploratory data analysis (EDA), feature engineering, model building with hyperparameter tuning, evaluation, and saving the trained model

---

## 📁 Project Structure

- breast-cancer.csv.       #csv dataset.
- knn_breast_cancer.ipynb.      # Main notebook
- knn_breast_cancer_model.joblib  # Saved best model
- scaler.joblib.               # Saved StandardScaler
- requirements.txt.           # Dependencies
-  README.md.

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- Joblib

---

## 🔽 Dataset

- **Source:** Breast Cancer Wisconsin Diagnostic Dataset (from `sklearn.datasets`)
- **Features:** 30 numeric features
- **Target:** 
  - `0` → Benign
  - `1` → Malignant

---

## 🚦 Workflow

### 1. 📚 Import Libraries

All standard data science libraries are imported including:
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report, confusion_matrix, RocCurveDisplay
import joblib


---

2. 📥 Load Dataset

we download the dataset from kaggle and then We load  the breast_cancer.csv dataset by pandas.


---

3. 🧹 Data Cleaning

Remove unwanted columns

Checked for missing/null values

checked for duplicated values 


---

4. 📊 Exploratory Data Analysis (EDA)

Basic info about dataset

Target distribution plotted using countplot


---
5.Train_Test_Split Data

Use 80% data for train and 20% data for model testing

---
6. 🛠️ Feature Engineering

Label encoding on target data.Put Benign =0 and Malignant=1

Standardization using StandardScaler to normalize feature scales


---

6. 🧠 Model Building & Hyperparameter Tuning

Base model: KNeighborsClassifier()

Hyperparameter tuning using GridSearchCV:

n_neighbors: 9,11,13


metric: euclidean, manhattan


Best model selected from GridSearch


---

7. 📈 Model Evaluation

Evaluation on test set using:

confusion_matrix

classification_report

ROC AUC Score and ROC Curve



---

8. 💾 Model Saving

Saved the trained model using joblib.dump():

joblib.dump(best_model, 'knn_breast_cancer_model.joblib')


---

📌 Results

Accuracy -98.25%
precision -100.00%
Recall -94.87%
ROC AUC Score	~0.98



---

✅ How to Run

1. Clone the repo:

git clone https://github.com/bijukar/knn_breast-cancer_classification.git
cd knn-breast-cancer


2. Install requirements:

pip install -r requirements.txt


3. Run the notebook:

jupyter notebook knn_breast_cancer.ipynb



