# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using supervised learning algorithms.

🌐 **Live Demo:** https://niteesh-fraud-detection.streamlit.app/

📂 **GitHub Repository:** https://github.com/alagandulaniteesh/credit-card-fraud-detection

## 📌 Project Overview

Credit card fraud is a major challenge in digital payments. This project uses Machine Learning to identify suspicious transactions and distinguish between legitimate and fraudulent activities.

The application allows users to upload transaction data and receive predictions, fraud probabilities, and downloadable results through an interactive Streamlit dashboard.

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Handle highly imbalanced datasets.
- Compare multiple Machine Learning algorithms.
- Improve model performance through SMOTE and hyperparameter tuning.
- Deploy the trained model using Streamlit Community Cloud.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib and Seaborn
- Joblib
- Streamlit

## 📊 Dataset

The project uses an anonymized credit card transaction dataset.

- Original transactions: 284,807
- Transactions after duplicate removal: 283,726
- Normal transactions: 283,253
- Fraudulent transactions: 473
- Features: 30 input features
- Target: `Class` (0 = Normal, 1 = Fraud)

The features include `Time`, `Amount`, and anonymized PCA components (`V1`–`V28`).

## 🤖 Machine Learning Models

The following models were evaluated:

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.6719 | 0.9658 |
| Decision Tree | 0.4961 | 0.9074 |
| Random Forest | 0.7991 | 0.9752 |
| SMOTE Random Forest | 0.8178 | 0.9752 |
| Tuned Random Forest | 0.8199 | 0.9773 |

## 🏆 Best Model Performance

**Tuned Random Forest with SMOTE**

- PR-AUC: 0.8199
- ROC-AUC: 0.9773
- Fraud Precision: 84%
- Fraud Recall: 80%
- Fraud F1-score: 82%

The final model is saved as:

`models/best_fraud_detection_pipeline.pkl`

## ⚙️ Project Workflow

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Train-test splitting
5. Model training and comparison
6. Handling class imbalance using SMOTE
7. Hyperparameter tuning
8. Model evaluation
9. Model serialization using Joblib
10. Streamlit deployment

## 📁 Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── create_sample.py
├── evaluate_sample.py
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_fraud_detection_pipeline.pkl
│
└── notebooks/
```

## 🚀 Run Locally

Clone the repository:

```bash
git clone https://github.com/alagandulaniteesh/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🌐 Live Application

Try the deployed application:

https://niteesh-fraud-detection.streamlit.app/

Upload a CSV containing the required 30 input features to generate fraud predictions.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes using an anonymized historical dataset. Its predictions should not be treated as a production banking fraud-detection system.

## 👨‍💻 Author

**Niteesh Kumar Alagandula**

GitHub: https://github.com/alagandulaniteesh

---

⭐ If you find this project interesting, consider giving the repository a star!