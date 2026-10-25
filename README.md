# Automated Credit Scoring System 💳

An end-to-end machine learning web application that evaluates applicant financial profiles to predict loan approval in real-time. Built with **Python**, **scikit-learn**, and **Streamlit**, the system uses a benchmarked Gaussian Naive Bayes classification model with standardized feature scaling and engineered non-linear inputs.

---

## 📌 Features

- **Real-Time Inference:** Evaluates applicant risk profiles instantly via an interactive web interface.
- **Engineered Feature Subsets:** Implements polynomial transformations (`DTI_Ratio_sq`, `Credit_Score_sq`) to capture complex decision boundaries.
- **Robust Preprocessing Pipeline:** Standardizes input variables using `StandardScaler`, converting annual income to monthly income and DTI percentages to decimal fractions to match training distributions.
- **Visual Risk Scoring:** Displays real-time approval confidence metrics, status alerts, and visual probability progress gauges.

---

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit
- **Machine Learning:** scikit-learn (GaussianNB, StandardScaler)
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 88.5% |
| **Recall** | 88.5% |
| **Precision** | 77.1% |
| **F1 Score** | 0.824 |

---
## 📸 Application Preview

<img width="1440" height="900" alt="Screenshot 2026-10-25 at 8 55 38 PM" src="https://github.com/user-attachments/assets/bbb3e62a-11a0-48f7-ab3b-bbddf707f375" />

<img width="1440" height="900" alt="Screenshot 2026-10-25 at 8 57 29 PM" src="https://github.com/user-attachments/assets/1fb34ede-7475-4822-9ca7-124dc2c03bed" />



