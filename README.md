# Fraud-detection
building a machine learning classifier to successfully detect fraud

# Financial Crime Detection: Optimising Recall in Imbalanced Datasets

## 1. Executive Summary
This project builds a Machine Learning classifier to detect fraudulent credit card transactions. 
The core challenge was the extreme class imbalance (only 0.17% of transactions were fraud). 
A baseline model predicting "Legit" for all transactions would achieve 99.8% accuracy but catch zero fraud.

I compared **XGBoost** and **Random Forest** to find the optimal balance between **Precision** (minimising false alarms) and **Recall** (catching fraud).

**Key Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, SMOTE (Imbalanced-Learn).

## 2. The Data
* **Source:** Real-world anonymised credit card transactions (European cardholders).
* **Volume:** ~284,000 transactions.
* **Complexity:** Highly imbalanced classes.
* **Preprocessing:** Applied **RobustScaler** to handle extreme outliers in transaction amounts (e.g., high-value VIP spending) that would otherwise skew standard scaling methods.

## 3. Handling Imbalance (SMOTE)
I used **SMOTE (Synthetic Minority Over-sampling Technique)** to synthesise new fraud examples for the training set.
* **Before SMOTE:** 394 Fraud cases (Too few for deep learning).
* **After SMOTE:** ~227,000 Fraud cases (Balanced 50/50 with legitimate transactions).
* *Note:* SMOTE was applied **only** to the training data to prevent data leakage into the test set.

## 4. Model Selection & Results
I benchmarked two powerful ensemble methods.

| Model | Recall (Fraud Caught) | Precision (False Alarm Rate) | Verdict |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | High | Very Low | Too simple; flagged too many innocent users. |
| **Random Forest** | 81% | **87%** | High precision, but missed 19% of fraud cases. |
| **XGBoost** | **86%** | 73% | **Champion Model.** Caught the most fraud (figures below). |
<img width="389" height="143" alt="image" src="https://github.com/user-attachments/assets/4213e5df-63e2-4a11-ab96-92463c1c397e" />


## 5. Business Conclusion
In a financial fraud context, a **False Negative** (missing fraud) is significantly more expensive than a **False Positive** (a temporary card block).

Therefore, I selected **XGBoost** as the production model. It achieved an **86% Recall rate**, ensuring the bank intercepts the maximum volume of fraudulent transactions, accepting a manageable trade-off in operational checks (73% Precision).
