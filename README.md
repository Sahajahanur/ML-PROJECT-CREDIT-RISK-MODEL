# ML-PROJECT-CREDIT-RISK-MODEL

A complete end-to-end Machine Learning project for predicting loan default risk.

## 🚀 Project Overview

Financial institutions like banks, NBFCs, and fintech companies face high financial risk from loan defaults.
This project builds a production-ready Credit Risk Prediction System using logistic regression with full feature engineering, IV–VIF based variable selection, hyperparameter tuning, and a deployed web application.

The final output is a Streamlit-based loan default risk calculator that predicts the probability of default for new applicants.

from EDA → Feature Engineering → IV/VIF → SMOTE → Hyperparameter Tuning → KS/AUC Evaluation → Deployment using Streamlit Cloud.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🎯 Business Objective

To predict whether a customer will default (1) or not default (0) based on their demographic, financial, and credit bureau information.

This helps lenders in:

* Credit decisioning
* Risk-based pricing
* Fraud detection
* Portfolio risk monitoring

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Key Highlights

* ✔ 50,000 rows × 30+ features (customers, loans, bureau data)
* ✔ Advanced data cleaning
* ✔ Feature engineering (LTI, delinquency ratio, avg DPD, etc.)
* ✔ Feature selection using VIF & Information Value (IV)
* ✔ SMOTE-Tomek for handling class imbalance
* ✔ Logistic Regression with Optuna hyperparameter tuning
* ✔ ROC–AUC, KS Statistic, Classification Report
* ✔ Model explainability using coefficients
* ✔ Full deployment using Streamlit Cloud
* ✔ Interactive app for real-time predictions

## ✅ FLOW DIAGRAM

    flowchart TD
    A[Raw Data] --> B[Preprocessing]
    
    B --> C[Business Rules]
    
    C --> D[Feature Engineering]
    
    D --> E[Feature Selection]
    
    E --> F[SMOTE + Tomek]
    
    F --> G[Logistic + Optuna]
    
    G --> H[Evaluation]
    
    H --> I[Deployment - Streamlit]


## 📂 Project Structure

ML-PROJECT-CREDIT-RISK-MODEL/

│
├── artifacts/

│   └── model_data.joblib            # Trained model + scaler + feature lists

│
├── main.py                          # Streamlit app

├── prediction_helper.py             # Prediction logic

├── requirements.txt                 # Python dependencies

├── README.md

└── .gitignore

## 📂 Dataset Summary

| File              | Description                             |
| ----------------- | --------------------------------------- |
| `customers.csv`   | Customer demographics                   |
| `loans.csv`       | Loan application details                |
| `bureau_data.csv` | Credit bureau history, DPD, loan months |

Target Variable:

📌 default → 1 = defaulter, 0 = non-defaulter

## 🛠️ Data Preprocessing

1️⃣ Merging Datasets

* customers.csv
* loans.csv
* bureau_data.csv
  
Merged on cust_id.

2️⃣ Missing Value Treatment

* Mode/median imputation
* Categorical cleaning (Personaal → Personal)

3️⃣ Outlier Treatment

* Business rules
   * Processing fee < 3% of loan amount
   * GST < 20%
   * Net disbursement ≤ sanctioned amount

4️⃣ Feature Engineering

Created new features:
* loan_to_income
* delinquency_ratio
* avg_dpd_per_delinquency
* Credit Utilization Ratio
* Total Loan Months
* One-hot encoding for categorical attributes

## 🧮 Feature Selection

✔ VIF (Variance Inflation Factor)

Removed multicollinear features:

* sanction_amount
* processing_fee
* gst
* net_disbursement
* principal_outstanding

✔ Information Value (IV)

Kept only predictive features (IV > 0.02).

## 📊 Modeling Approach

Algorithm Used:

Logistic Regression (chosen for interpretability + auditability in finance)

## ⚖️ Handling Class Imbalance

Used SMOTE-Tomek to oversample minority class & clean overlapping boundaries.

## Hyperparameter Tuning

Used Optuna with:

* Solver
* C
* Tolerance
* Class weight

## 🤖 Model Training

Model used: Logistic Regression

* Tuned using Optuna
* Metrics optimized: Macro F1-score
* Max iterations: 10,000

## 📈 Model Evaluation

| Metric       | Score                 |
| ------------ | --------------------- |
| AUC          | 0.92+                 |
| KS Statistic | ~0.71                 |
| Accuracy     | High                  |
| Recall       | Strong for defaulters |
| F1 Score     | Balanced              |

## 🧪 KS Statistic (Kolmogorov–Smirnov Test)

Measures separation between "good" and "bad" populations.

KS = max(Cumulative Good Rate − Cumulative Bad Rate)
Higher KS → better risk discriminatory power.


## 🧮 Final Logistic Regression Formula

Extracted using:

    final_model.coef_
    final_model.intercept_

## 🌐 Live App Demo

👉 Streamlit App:

🔗 https://ml-project-credit-risk-model-srl.streamlit.app/

<img width="947" height="735" alt="image" src="https://github.com/user-attachments/assets/70cbf2ae-6f50-49de-85cb-9ef7a97c16ca" />

## 🧠 App Features

* 13 feature inputs
* Automatic Loan to Income calculation
* Real-time inference
* Risk categorization
* Clean & modern UI

## 🗂️ Folder Structure

📦 ML-PROJECT-CREDIT-RISK-MODEL

 ┣ 📂 artifacts
 
 ┃ ┣ model_data.joblib
 
 ┃ ┗ scaler.joblib
 
 ┣ 📂 notebooks
 
 ┣ main.py
 
 ┣ prediction_helper.py
 
 ┣ requirements.txt
 
 ┣ README.md

 ## ▶️ Run Locally

 1️⃣ Clone repo

     git clone https://github.com/your-username/ML-PROJECT-CREDIT-RISK-MODEL.git
    cd ML-PROJECT-CREDIT-RISK-MODEL

  2️⃣ Install dependencies

    pip install -r requirements.txt

  3️⃣ Run Streamlit  

    streamlit run main.py

## 🧪 Tech Stack

* Python
* Pandas, NumPy
* Imbalanced-learn (SMOTE)
* Statsmodels (VIF)
* Optuna
* Streamlit
* Joblib

## 📘 Learning Outcomes

* Real-world risk analytics
* Credit scoring model development
* End-to-end ML pipeline
* Model deployment
* Understanding of feature engineering in finance
* ROC, KS, IV, VIF — industry standards

##  📬 Contacts  

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:connectingsrl@gmail.com)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sahajahanur-laskar/)




