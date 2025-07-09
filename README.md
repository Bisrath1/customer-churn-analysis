# 📊 Customer Churn Analysis and Prediction

**Machine Learning Internship Project – Saiket Systems**

This repository contains a complete machine learning pipeline to analyze and predict customer churn in a telecommunications company. The project was developed as part of a machine learning internship at **Saiket Systems**.

---

## 🚀 Objective

The goal of this project is to:

* Analyze key factors contributing to customer churn
* Build predictive models to identify high-risk customers
* Deliver actionable business recommendations to reduce churn

---

## 🧠 Project Workflow

```
📁 customer-churn-analysis/
├── data/            # Raw dataset
├── notebooks/       # Test scripts and visualizations
├── outputs/         # Saved models, plots, reports
├── src/             # Source code for each task
├── README.md        # Project overview
└── requirements.txt # Python dependencies
```

---

## ✅ Tasks Overview

### 📂 Task 1: Data Preparation

* Loaded the Telco Customer Churn dataset
* Cleaned missing values
* Converted categorical variables into numerical format

> 📄 `src/data_prep.py`

---

### 📂 Task 2: Data Splitting

* Split data into training (80%) and testing (20%) sets
* Ensured target class balance using stratification

> 📄 `src/data_split.py`

---

### 📂 Task 3: Feature Selection

* Selected top features using univariate statistical tests (e.g., SelectKBest)
* Identified impactful features such as `tenure`, `MonthlyCharges`, and `ContractType`

> 📄 `src/feature_select.py`

---

### 📂 Task 4: Model Selection

* Tested multiple classifiers: Logistic Regression, Decision Trees, Random Forest
* Chose **Random Forest** for its balance between performance and interpretability

> 📄 `src/train_model.py`

---

### 📂 Task 5: Evaluation & Visualization

* Evaluated the model using accuracy, precision, recall, and AUC
* Visualized results:

  * 📊 Churn distribution
  * 🔥 Correlation heatmap
  * ✅ Confusion matrix
  * 📈 ROC Curve

> 📄 `src/evaluate_model.py`
> 📄 `notebooks/05_visualization.py`

---

### 📂 Task 6: Final Recommendations

* Generated a summary report with:

  * Key insights from the model
  * Business recommendations
  * Feature importance highlights

> 📄 `notebooks/06_final_recommendations.py`

---

## 📊 Final Model Results

* **Model**: Random Forest
* **Accuracy**: \~85%
* **Precision (Churn)**: \~79%
* **Recall (Churn)**: \~72%
* **AUC**: \~0.91

---

## 📌 How to Run

1. Clone the repo

```bash
git clone https://github.com/Bisrath1/customer-churn-analysis.git
cd customer-churn-analysis
```

2. Set up virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Run any notebook in the `notebooks/` directory or individual Python scripts in `src/`.

---

## 🏆 Acknowledgment

This project was completed during the **Machine Learning Internship at Saiket Systems**.
It showcases real-world ML workflows, from raw data to model deployment recommendations.

---

