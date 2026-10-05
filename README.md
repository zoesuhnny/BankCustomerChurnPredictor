# Bank Customer Churn Predictor

A PyTorch neural network that predicts whether a bank customer will churn,
trained on an imbalanced dataset (~20% churned).

> **Note:** This was a learning project to practice building and evaluating
> neural networks in PyTorch. It isn't polished, so some code may be irrelevant.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zoesuhnny/BankCustomerChurnPredictor/blob/main/CustomersChurnPrediction.ipynb)

## Overview
- **Data:** [Bank Customers dataset](https://www.kaggle.com/datasets/santoshd3/bank-customers) (Kaggle), 10,000 customers, 11 features
- **Model:** 2-layer feed-forward network (11 → 4 → 1, ReLU)
- **Training:** BCEWithLogitsLoss + Adam, standardized features, 80/20 train/test split
- **Evaluation:** precision, recall, F1, and ROC-AUC (≈0.85), since accuracy alone
  is misleading on imbalanced data

## Next steps
- Address class imbalance (weighted loss, stratified split, threshold tuning)
- Improve feature encoding (one-hot Geography, drop Surname)
- Compare against simpler baselines like logistic regression

## Built with
Python, PyTorch, Pandas, NumPy, scikit-learn, Matplotlib, Google Colab
